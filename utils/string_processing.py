

def ensure_terminator(path, terminator="/"):
    path = path.replace('\\', '/')
    if path == "":
        return path
    if path.endswith(terminator):
        return path
    return path + terminator