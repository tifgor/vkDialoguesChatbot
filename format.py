def format(s):

    if len(s.split()) > 20:
        s = " ".join(s.split()[:20])
    if "?" or "!" or "." or ","  or ";" or "*" or "/" or "\\" or "*" or ")" or "(" in s:
        punct = False
        s1 = ""
        for i in range(len(s)):
            if s[i] == "(" or s[i] == ")" or s[i] == "!" or s[i] == "." or s[i] == "," \
                    or s[i] == ":" or s[i] == ";" or s[i] == "?" or s[i] == "*" or s[i] == "\\":
                if punct == False and i != 0:
                    punct = True
                    s1 = s1 + " "
                    s1 = s1 + s[i]
            else:
                punct = False
                s1 = s1 + s[i]
        return s1
    else:
        return s
