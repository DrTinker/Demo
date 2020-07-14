import re
import unicodedata


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)

    s = re.sub(r'([\，.\@.\*.\#.\、.\！.\？.\?.\!.\$.\%.])', r'', s)
    s = re.sub(r'([\d])', r'', s)
    s = re.sub(r'\s+', r'', s)

    return s


def is_good_line(line):
    if len(line) == 0 or len(line) < 5:
        return False
    ch_count = 0
    for c in line:
        # 中文字符范围
        if '\u4e00' <= c <= '\u9fff':
            ch_count += 1
    if ch_count / float(len(line)) >= 0.8 and len(re.findall(r'[a-zA-Z0-9]',
                                                  ''.join(line))) < 3 \
            and len(re.findall(r'[ˇˊˋˍεπのゞェーω]', ''.join(line))) < 3:
        return True
    return False


def regular(s):
    # 统一过长的省略号等符号
    s = s.replace('/', '')
    s = re.sub(r'…{1,100}', '…', s)
    s = re.sub(r'\.{3,100}', '…', s)
    s = re.sub(r'···{2,100}', '…', s)
    s = re.sub(r',{1,100}', '，', s)
    s = re.sub(r'\.{1,100}', '。', s)
    s = re.sub(r'。{1,100}', '。', s)
    s = re.sub(r'\?{1,100}', '？', s)
    s = re.sub(r'？{1,100}', '？', s)
    s = re.sub(r'!{1,100}', '！', s)
    s = re.sub(r'！{1,100}', '！', s)
    s = re.sub(r'~{1,100}', '～', s)
    s = re.sub(r'～{1,100}', '～', s)
    s = re.sub(r'[“”]{1,100}', '"', s)
    s = re.sub('[^\w\u4e00-\u9fff"。，？！～·]+', '', s)
    s = re.sub(r'[ˇˊˋˍεπのゞェーω]', '', s)

    return s
