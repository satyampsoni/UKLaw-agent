"""
XML Parser for legislation.gov.uk Acts.

Converts the official XML representation of UK Acts into our
LegalNode tree. This is where the raw government data becomes our
structured PageIndex model.

XML Structure (from legislation.gov.uk):
    <Legislation>
        <Primary>
            <Body>
                <Part id="part-1">
                    <Number><Strong>PART 1</Strong></Number>
                    <Title>Preliminary</Title>
                    <Chapter id="part-1-chapter-1">     ← optional
                        <Pblock id="...crossheading..."> ← optional
                            <Title>...</Title>
                            <P1group>
                                <Title>Section title</Title>
                                <P1 id="section-1">
                                    <Pnumber>1</Pnumber>
                                    <P1para>
                                        <P2 id="section-1-1">  ← subsection
                                            <Pnumber>1</Pnumber>
                                            <P2para>
                                                <Text>...</Text>
                                                <P3>            ← clause (a), (b)...
                                                    <P4>        ← sub-clause (i), (ii)...

The parser does a depth-first walk, building LegalNode objects with
parent_id references to form our tree. It strips out inline markup
(<Addition>, <Substitution>, <CommentaryRef>) to get clean text.

Key design choices:
  1. We use lxml (not stdlib xml.etree) because lxml handles the large
     legislation.gov.uk XML files more efficiently and has better XPath.
  2. Each node gets a citation like "DPA 2018, s.6(1)(a)" built up as
     we walk down the tree — this is how lawyers actually reference law.
  3. We extract text from ALL <Text> descendants but ignore commentary
     metadata — the parser focuses on the operative legal text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

from app.models.legal_document import (
    DocumentSource,
    DocumentStatus,
    LegalDocument,
    LegalNode,
    NodeType,
    NODE_DEPTH,
)


# ────────────────────────────────────────────────────────────────────
# Parse context (tracks state during tree walk)
# ────────────────────────────────────────────────────────────────────

@dataclass
class ParseContext:
    """Carries metadata and state down through the recursive parse."""
    document_id: str
    short_title: str          # e.g. "DPA 2018" — for citations
    full_title: str           # e.g. "Data Protection Act 2018"
    year: int
    base_url: str             # e.g. "https://www.legislation.gov.uk/ukpga/2018/12"
    nodes: list[LegalNode] = field(default_factory=list)
    _order_counter: int = 0

    def next_order(self) -> int:
        """Global ordering so nodes stay in document order."""
        idx = self._order_counter
        self._order_counter += 1
        return idx


# ────────────────────────────────────────────────────────────────────
# Text extraction helpers
# ────────────────────────────────────────────────────────────────────

def _strip_ns(tag: str) -> str:
    """Remove namespace prefix from element tag.

    legislation.gov.uk XML sometimes uses namespaces,
    sometimes doesn't. This handles both.
    """
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _get_text(element: etree._Element) -> str:
    """Extract clean text from an element, stripping inline markup.

    <Text>Part 3 makes provision about
        <CommentaryRef Ref="key-123"/>
        the processing of <Substitution>personal data</Substitution>
    </Text>

    → "Part 3 makes provision about the processing of personal data"

    We iterate ALL text and tail content to handle mixed content elements.
    """
    parts: list[str] = []
    # Walk the element tree and collect all text
    for node in element.iter():
        tag = _strip_ns(node.tag) if isinstance(node.tag, str) else ""
        # Skip entire Commentary and Metadata subtrees
        if tag in ("CommentaryRef", "FootnoteRef"):
            continue
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)

    text = " ".join(parts)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_direct_text(element: etree._Element) -> str:
    """Get text from direct <Text> children only (not nested P2/P3/etc)."""
    texts: list[str] = []
    for child in element:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "Text":
            texts.append(_get_text(child))
    return " ".join(texts)


def _get_para_text(para_element: etree._Element) -> str:
    """Extract text from a P1para, P2para, P3para, etc.

    Only gets direct <Text> children — sub-provisions (P2, P3, P4)
    are handled by the recursive parser, not here.
    """
    return _get_direct_text(para_element)


def _get_number(element: etree._Element) -> str:
    """Extract the provision number from a <Number> or <Pnumber> element.

    P2/P3/P4 Pnumbers have plain text: <Pnumber>1</Pnumber>
    P1 Pnumbers often have NO text, only <CommentaryRef> children.
    In that case, we fall back to extracting the number from the
    element's `id` attribute (e.g. "section-3" → "3").
    """
    for child in element:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag in ("Number", "Pnumber"):
            # Get text, filtering out CommentaryRef content
            raw_text = (child.text or "").strip()
            if not raw_text:
                # Check <Strong> or other text-bearing children
                for sub in child:
                    sub_tag = _strip_ns(sub.tag) if isinstance(sub.tag, str) else ""
                    if sub_tag in ("Strong", "Emphasis"):
                        raw_text = _get_text(sub).strip()
                        break
            if raw_text:
                return raw_text
    # Fallback: extract number from the id attribute
    xml_id = element.attrib.get("id", "")
    if xml_id:
        # "section-3" → "3", "section-3-1" → "3" (for P1 only)
        match = re.search(r"section-(\d+[A-Z]*)$", xml_id)
        if match:
            return match.group(1)
        # "section-3-1" → won't match above, that's for P2
    return ""


def _get_title(element: etree._Element) -> str:
    """Extract the title from a <Title> child element."""
    for child in element:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "Title":
            return _get_text(child).strip()
    return ""


# ────────────────────────────────────────────────────────────────────
# Node builders for each level
# ────────────────────────────────────────────────────────────────────

def _make_node(
    ctx: ParseContext,
    node_type: NodeType,
    node_id: str,
    parent_id: Optional[str],
    title: str,
    text: str,
    citation: str,
    hierarchy_path: str,
    source_url: str = "",
) -> LegalNode:
    """Create a LegalNode and add it to the context.

    All IDs are prefixed with the document_id to ensure global uniqueness.
    legislation.gov.uk uses the same id format across Acts (e.g. 'section-151'
    exists in both DPA 2018 and OSA 2023), so we prefix to avoid collisions.
    """
    # Prefix IDs with document_id for global uniqueness
    # Skip if already prefixed (e.g. the root ACT node IS the document_id)
    if node_id and not node_id.startswith(ctx.document_id):
        node_id = f"{ctx.document_id}:{node_id}"
    if parent_id and not parent_id.startswith(ctx.document_id):
        parent_id = f"{ctx.document_id}:{parent_id}"

    node = LegalNode(
        id=node_id,
        document_id=ctx.document_id,
        node_type=node_type,
        depth=NODE_DEPTH.get(node_type, 0),
        parent_id=parent_id,
        order_index=ctx.next_order(),
        hierarchy_path=hierarchy_path,
        title=title,
        text=text,
        citation=citation,
        source=DocumentSource.LEGISLATION_GOV_UK,
        source_url=source_url or f"{ctx.base_url}/{node_id.replace('-', '/')}",
        year=ctx.year,
    )
    ctx.nodes.append(node)
    return node


# ────────────────────────────────────────────────────────────────────
# Recursive parsers for each level
# ────────────────────────────────────────────────────────────────────

def _parse_clause(
    el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
    parent_citation: str,
) -> None:
    """Parse P3 (clause) and P4 (sub-clause) elements.

    P3 has <Pnumber>(a)</Pnumber> and <P3para>
    P4 has <Pnumber>(i)</Pnumber> and <P4para>
    """
    tag = _strip_ns(el.tag)
    xml_id = el.attrib.get("id", "")
    number = _get_number(el)

    # Find the para element (P3para or P4para)
    para_tag = f"{tag}para"
    text_parts: list[str] = []
    for child in el:
        child_tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if child_tag == para_tag:
            text_parts.append(_get_para_text(child))

    text = " ".join(text_parts).strip()
    citation = f"{parent_citation}({number.strip('()')})" if number else parent_citation
    title = f"Clause {number}" if number else "Clause"
    path = f"{parent_path} > {title}"

    node = _make_node(
        ctx=ctx,
        node_type=NodeType.CLAUSE,
        node_id=xml_id or f"{parent_id}-clause-{number}",
        parent_id=parent_id,
        title=title,
        text=text,
        citation=citation,
        hierarchy_path=path,
    )

    # Recurse into nested P4 inside P3para, or deeper
    for child in el:
        child_tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if child_tag == para_tag:
            for sub in child:
                sub_tag = _strip_ns(sub.tag) if isinstance(sub.tag, str) else ""
                if sub_tag in ("P3", "P4", "P5"):
                    _parse_clause(sub, ctx, node.id, path, citation)


def _parse_subsection(
    el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
    parent_citation: str,
) -> None:
    """Parse a <P2> (subsection) element.

    Structure: <P2 id="section-1-1">
                   <Pnumber>1</Pnumber>
                   <P2para>
                       <Text>...</Text>
                       <P3>...</P3>  ← clauses
    """
    xml_id = el.attrib.get("id", "")
    number = _get_number(el)

    # Get direct text from P2para (not from nested P3/P4)
    text_parts: list[str] = []
    for child in el:
        child_tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if child_tag == "P2para":
            text_parts.append(_get_para_text(child))
    text = " ".join(text_parts).strip()

    citation = f"{parent_citation}({number})" if number else parent_citation
    title = f"Subsection ({number})" if number else "Subsection"
    path = f"{parent_path} > ({number})" if number else f"{parent_path} > Subsection"

    node = _make_node(
        ctx=ctx,
        node_type=NodeType.SUBSECTION,
        node_id=xml_id or f"{parent_id}-sub-{number}",
        parent_id=parent_id,
        title=title,
        text=text,
        citation=citation,
        hierarchy_path=path,
    )

    # Recurse into clauses (P3) inside P2para
    for child in el:
        child_tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if child_tag == "P2para":
            for sub in child:
                sub_tag = _strip_ns(sub.tag) if isinstance(sub.tag, str) else ""
                if sub_tag in ("P3", "P4"):
                    _parse_clause(sub, ctx, node.id, path, citation)


def _parse_section(
    p1_el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
    section_title: str,
) -> None:
    """Parse a <P1> (section) element.

    Structure: <P1group>
                   <Title>Section title</Title>
                   <P1 id="section-1">
                       <Pnumber>1</Pnumber>
                       <P1para>
                           <P2>...</P2>  ← subsections
                           <Text>...</Text>  ← direct text (if no subsections)

    The title comes from the parent <P1group>, passed as section_title.
    """
    xml_id = p1_el.attrib.get("id", "")
    number = _get_number(p1_el)

    # Build citation: "DPA 2018, s.6"
    citation = f"{ctx.short_title}, s.{number}" if number else ctx.short_title

    title = section_title or f"Section {number}"
    path = f"{parent_path} > s.{number} {title}" if number else f"{parent_path} > {title}"

    # Get direct text from P1para (not from nested P2)
    text_parts: list[str] = []
    for child in p1_el:
        child_tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if child_tag == "P1para":
            text_parts.append(_get_para_text(child))
    text = " ".join(text_parts).strip()

    node = _make_node(
        ctx=ctx,
        node_type=NodeType.SECTION,
        node_id=xml_id or f"section-{number}",
        parent_id=parent_id,
        title=title,
        text=text,
        citation=citation,
        hierarchy_path=path,
    )

    # Recurse into subsections (P2) inside P1para
    for child in p1_el:
        child_tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if child_tag == "P1para":
            for sub in child:
                sub_tag = _strip_ns(sub.tag) if isinstance(sub.tag, str) else ""
                if sub_tag == "P2":
                    _parse_subsection(sub, ctx, node.id, path, citation)
                elif sub_tag in ("P3", "P4"):
                    _parse_clause(sub, ctx, node.id, path, citation)


def _parse_p1group(
    el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
) -> None:
    """Parse a <P1group> which wraps one or more <P1> sections."""
    section_title = _get_title(el)
    for child in el:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "P1":
            _parse_section(child, ctx, parent_id, parent_path, section_title)


def _parse_crossheading(
    el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
) -> None:
    """Parse a <Pblock> (cross-heading group).

    Cross-headings group related sections under a topic.
    For example: "Lawfulness of processing" groups s.6-s.11.
    """
    xml_id = el.attrib.get("id", "")
    title = _get_title(el) or "Cross-heading"
    path = f"{parent_path} > {title}"

    node = _make_node(
        ctx=ctx,
        node_type=NodeType.CROSSHEADING,
        node_id=xml_id or f"{parent_id}-xh-{title[:20]}",
        parent_id=parent_id,
        title=title,
        text="",  # Cross-headings have no body text
        citation=f"{ctx.short_title}, {title}",
        hierarchy_path=path,
    )

    # Process child P1groups and nested Pblocks
    for child in el:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "P1group":
            _parse_p1group(child, ctx, node.id, path)
        elif tag == "P1":
            # Some Pblocks contain P1 directly without P1group
            section_title = _get_title(child)
            _parse_section(child, ctx, node.id, path, section_title)
        elif tag == "Pblock":
            # Nested cross-headings
            _parse_crossheading(child, ctx, node.id, path)


def _parse_chapter(
    el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
) -> None:
    """Parse a <Chapter> element."""
    xml_id = el.attrib.get("id", "")
    number = _get_title_number(el)
    title = _get_title(el) or f"Chapter {number}"
    path = f"{parent_path} > Chapter {number}" if number else f"{parent_path} > {title}"

    node = _make_node(
        ctx=ctx,
        node_type=NodeType.CHAPTER,
        node_id=xml_id or f"{parent_id}-ch-{number}",
        parent_id=parent_id,
        title=title,
        text="",
        citation=f"{ctx.short_title}, {title}",
        hierarchy_path=path,
    )

    for child in el:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "Pblock":
            _parse_crossheading(child, ctx, node.id, path)
        elif tag == "P1group":
            _parse_p1group(child, ctx, node.id, path)
        elif tag == "P1":
            _parse_section(child, ctx, node.id, path, _get_title(child))


def _parse_schedule(
    el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
) -> None:
    """Parse a <Schedule> element.

    Schedules are like appendices — they contain Parts, Pblocks,
    and paragraphs (P1 at schedule level = paragraph, not section).
    """
    xml_id = el.attrib.get("id", "")
    number = _get_title_number(el)
    title = _get_title(el) or f"Schedule {number}"
    full_title = f"Schedule {number}" if number else title
    path = f"{parent_path} > {full_title}"

    node = _make_node(
        ctx=ctx,
        node_type=NodeType.SCHEDULE,
        node_id=xml_id or f"schedule-{number}",
        parent_id=parent_id,
        title=f"{full_title}: {title}" if title != full_title else full_title,
        text="",
        citation=f"{ctx.short_title}, Sch.{number}",
        hierarchy_path=path,
    )

    for child in el:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "Part":
            _parse_part(child, ctx, node.id, path, is_schedule=True)
        elif tag == "Pblock":
            _parse_crossheading(child, ctx, node.id, path)
        elif tag == "P1group":
            _parse_p1group(child, ctx, node.id, path)
        elif tag == "P1":
            _parse_section(child, ctx, node.id, path, _get_title(child))


def _parse_part(
    el: etree._Element,
    ctx: ParseContext,
    parent_id: str,
    parent_path: str,
    is_schedule: bool = False,
) -> None:
    """Parse a <Part> element."""
    xml_id = el.attrib.get("id", "")
    number = _get_title_number(el)
    title = _get_title(el) or f"Part {number}"
    path = f"{parent_path} > Part {number}" if number else f"{parent_path} > {title}"

    node = _make_node(
        ctx=ctx,
        node_type=NodeType.PART,
        node_id=xml_id or f"part-{number}",
        parent_id=parent_id,
        title=title,
        text="",
        citation=f"{ctx.short_title}, Part {number}",
        hierarchy_path=path,
    )

    for child in el:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "Chapter":
            _parse_chapter(child, ctx, node.id, path)
        elif tag == "Pblock":
            _parse_crossheading(child, ctx, node.id, path)
        elif tag == "P1group":
            _parse_p1group(child, ctx, node.id, path)
        elif tag == "P1":
            _parse_section(child, ctx, node.id, path, _get_title(child))


def _get_title_number(el: etree._Element) -> str:
    """Get the number from a <Number> child, e.g. 'PART 1' → '1'."""
    for child in el:
        tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
        if tag == "Number":
            raw = _get_text(child).strip()
            # Extract just the number from "PART 1", "CHAPTER 2", "SCHEDULE 3"
            match = re.search(r"(\d+[A-Z]*)", raw)
            return match.group(1) if match else raw
    return ""


# ────────────────────────────────────────────────────────────────────
# Main public API
# ────────────────────────────────────────────────────────────────────

def parse_act_xml(
    xml_bytes: bytes,
    document_id: str,
    short_title: str,
    full_title: str,
    year: int,
    base_url: str,
) -> LegalDocument:
    """
    Parse a full Act XML from legislation.gov.uk into a LegalDocument.

    Args:
        xml_bytes:   Raw XML bytes from the API
        document_id: Our internal slug, e.g. "dpa-2018"
        short_title: For citations, e.g. "DPA 2018"
        full_title:  Full Act name, e.g. "Data Protection Act 2018"
        year:        Year of the Act
        base_url:    legislation.gov.uk URL base

    Returns:
        LegalDocument with a flat list of hierarchical LegalNode objects.
    """
    # Parse XML — use recover=True to handle any malformed content
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    root = etree.fromstring(xml_bytes, parser)

    ctx = ParseContext(
        document_id=document_id,
        short_title=short_title,
        full_title=full_title,
        year=year,
        base_url=base_url,
    )

    # Create the root ACT node
    act_node = _make_node(
        ctx=ctx,
        node_type=NodeType.ACT,
        node_id=document_id,
        parent_id=None,
        title=full_title,
        text="",
        citation=short_title,
        hierarchy_path=full_title,
        source_url=base_url,
    )

    # Find the <Body> element — it contains all Parts/Chapters/Sections
    # Walk through: <Legislation> → <Primary> → <Body>
    body = None
    schedules_container = None
    for el in root.iter():
        tag = _strip_ns(el.tag) if isinstance(el.tag, str) else ""
        if tag == "Body" and body is None:
            body = el
        if tag == "Schedules" and schedules_container is None:
            schedules_container = el

    if body is not None:
        for child in body:
            tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
            if tag == "Part":
                _parse_part(child, ctx, act_node.id, full_title)
            elif tag == "Chapter":
                _parse_chapter(child, ctx, act_node.id, full_title)
            elif tag == "Pblock":
                _parse_crossheading(child, ctx, act_node.id, full_title)
            elif tag == "P1group":
                _parse_p1group(child, ctx, act_node.id, full_title)

    # Parse Schedules (they sit outside <Body>, inside <Schedules>)
    if schedules_container is not None:
        for child in schedules_container:
            tag = _strip_ns(child.tag) if isinstance(child.tag, str) else ""
            if tag == "Schedule":
                _parse_schedule(child, ctx, act_node.id, full_title)

    return LegalDocument(
        document_id=document_id,
        title=full_title,
        source=DocumentSource.LEGISLATION_GOV_UK,
        source_url=base_url,
        year=year,
        nodes=ctx.nodes,
    )
