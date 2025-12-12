import re

def split_by_length(text: str, max_len: int) -> list[str]:
    chunks: list[str] = []
    pos = 0       

    while pos < len(text):
        hard_end = min(pos + max_len, len(text))
        if hard_end == len(text):
            chunks.append(text[pos:])
            break

        soft_end = text.rfind(' ', pos, hard_end)
        cut_point = soft_end if soft_end > pos else hard_end
        chunks.append(text[pos:cut_point])

        pos = cut_point
        while pos < len(text) and text[pos] == ' ':
            pos += 1
    return chunks

def chunk_text_code_aware(text: str, *, max_chunk_size: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    chunks = []
    current_chunk = []
    current_length = 0
    
    in_code_block = False
    current_language = ""
    
    fence_pattern = re.compile(r'^ *`{3,}(.*)$') # Code fence + language
    for line in lines:
        line_length = len(line)
        
        match = fence_pattern.match(line.strip())
        is_fence_toggle = bool(match)
        closing_overhead = len("\n```") if in_code_block and not is_fence_toggle else 0
        
        if current_chunk and (current_length + line_length + closing_overhead > max_chunk_size):
            if in_code_block:
                current_chunk.append("\n```")
                chunks.append("".join(current_chunk))
                start_tag = f"```{current_language}\n"
                current_chunk = [start_tag]
                current_length = len(start_tag)
            else:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_length = 0

        current_chunk.append(line)
        current_length += line_length
        
        if is_fence_toggle:
            if in_code_block:
                in_code_block = False
                current_language = ""
            else:
                in_code_block = True
                current_language = match.group(1).strip()

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks