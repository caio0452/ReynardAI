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

def balance_code_fences(raw_chunks: list[str]) -> list[str]:
    balanced: list[str] = []
    inside_code = False      
    open_fence_line = "" 

    for chunk in raw_chunks:
        out = ""
        if inside_code:
            out += open_fence_line + "\n"
        out += chunk

        tmp_is_in_code = inside_code
        for line in chunk.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("```"):
                tmp_state = not tmp_is_in_code
                if tmp_state:
                    open_fence_line = stripped.rstrip('\n')

        if tmp_is_in_code:
            if not out.endswith("\n"):
                out += "\n"
            out += "```"
        balanced.append(out)
        inside_code = tmp_is_in_code
    return balanced