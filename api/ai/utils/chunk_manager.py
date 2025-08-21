import re
import html
from typing import List, Dict

class HTMLChunker:
    def __init__(self):
        self.TAG_RE = re.compile(r'(<[^>]+>)', re.DOTALL)
        self.ENTITY_RE = re.compile(r'&[A-Za-z0-9#]+;')
        self.BLOCK_BREAK_TAGS = {
            "p", "div", "br", "li", "ul", "ol",
            "h1", "h2", "h3", "h4", "h5", "h6",
            "table", "tr", "th", "td",
        }
        self._SENT_END_CHARS = ".!?;؟؛…。！？；．।॥։።፧"
        self._CLOSERS = "’”\"'»›）)]】》〗〙〛〉］｝』」"
        self._OPTIONAL_CLOSERS_RE = f"[{re.escape(self._CLOSERS)}]*"
        self._TRAILING_CLOSE_TAGS_RE = r"(?:\s*(?:</[^>]+>))*\s*$"
        self._COMPLETE_SENTENCE_AT_END = re.compile(
            rf"(?:\.{{3}}|[{re.escape(self._SENT_END_CHARS)}]){self._OPTIONAL_CLOSERS_RE}{self._TRAILING_CLOSE_TAGS_RE}"
        )

    def _tag_name(self, tag: str) -> str:
        m = re.match(r'<\s*/?\s*([a-zA-Z0-9]+)', tag)
        return m.group(1).lower() if m else ""

    def _iter_html_tokens(self, html_src: str):
        pos = 0
        for m in self.TAG_RE.finditer(html_src):
            if m.start() > pos:
                yield ("text", html_src[pos:m.start()])
            yield ("tag", m.group(1))
            pos = m.end()
        if pos < len(html_src):
            yield ("text", html_src[pos:])

    def _iter_text_units(self, raw_text: str):
        i = 0
        while i < len(raw_text):
            em = self.ENTITY_RE.match(raw_text, i)
            if em:
                raw = em.group(0)
                try:
                    plain = html.unescape(raw)
                except Exception:
                    plain = raw
                yield raw, plain
                i = em.end()
            else:
                ch = raw_text[i]
                yield ch, ch
                i += 1

    def chunk_html_streaming(self, html_src: str, max_text_chars: int = 1000) -> List[Dict[str, str]]:
        chunks: List[Dict[str, str]] = []
        html_buf: List[str] = []
        text_buf: List[str] = []
        cur_len = 0

        def flush():
            nonlocal html_buf, text_buf, cur_len
            if html_buf or text_buf:
                chunks.append({
                    "html": "".join(html_buf),
                    "text": "".join(text_buf).strip()
                })
            html_buf = []
            text_buf = []
            cur_len = 0

        for kind, token in self._iter_html_tokens(html_src):
            if kind == "tag":
                html_buf.append(token)
                name = self._tag_name(token)
                if name == "br" or token.startswith("</"):
                    if name in self.BLOCK_BREAK_TAGS or name == "br":
                        text_buf.append("\n")
                        cur_len += 1
                continue
            for raw_unit, plain_unit in self._iter_text_units(token):
                if cur_len + len(plain_unit) > max_text_chars and cur_len > 0:
                    flush()
                html_buf.append(raw_unit)
                text_buf.append(plain_unit)
                cur_len += len(plain_unit)
        flush()
        return chunks

    def get_incomplete_end(self, chunk_html: str, backtrack: int = 300):
        s = chunk_html or ""
        n = len(s)
        if n == 0:
            return "", ""
        last_lt = s.rfind("<")
        last_gt = s.rfind(">")
        tag_idx = last_lt if (last_lt > last_gt) else None
        wstart = max(0, n - backtrack)
        last_amp = s.rfind("&", wstart, n)
        last_semi = s.rfind(";", wstart, n)
        entity_idx = last_amp if (last_amp > last_semi) else None
        if self._COMPLETE_SENTENCE_AT_END.search(s):
            sentence_idx = None
        else:
            window = s[wstart:n]
            ell = window.rfind("...")
            if ell >= 0:
                cut = wstart + ell + 3
            else:
                last_punct = -1
                for ch in self._SENT_END_CHARS:
                    j = window.rfind(ch)
                    if j > last_punct:
                        last_punct = j
                if last_punct >= 0:
                    cut = wstart + last_punct + 1
                else:
                    sentence_idx = wstart
                    cut = None
            if cut is not None:
                m = re.match(self._OPTIONAL_CLOSERS_RE, s[cut:])
                if m and m.end() > 0:
                    cut += m.end()
                m2 = re.match(r"(?:\s*(?:</[^>]+>))*\s*", s[cut:])
                if m2 and m2.end() > 0:
                    cut += m2.end()
                sentence_idx = cut if cut < n else None
        candidates = [idx for idx in (tag_idx, entity_idx, sentence_idx) if isinstance(idx, int) and 0 <= idx < n]
        if not candidates:
            return s, ""
        start = min(candidates)
        return s[:start], s[start:]

class ChunkPipeline:
    def __init__(self, max_text_chars=1000, backtrack=300):
        self.chunker = HTMLChunker()
        self.max_text_chars = max_text_chars
        self.backtrack = backtrack

    def process(self, html_src):
        chunks = self.chunker.chunk_html_streaming(html_src, self.max_text_chars)
        results = []
        for chunk in chunks:
            head, tail = self.chunker.get_incomplete_end(chunk["html"], self.backtrack)
            results.append({
                "html": chunk["html"],
                "text": chunk["text"],
                "head": head,
                "tail": tail,
            })
        return results