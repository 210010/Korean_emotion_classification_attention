from konlpy import Okt as Twitter
import re
def tokenization(x_val):
    tmp = []
    max_count = 0
    set_words = set()
    twitter = Twitter()

    for i,x in enumerate(x_val):
        x = re.sub(r"[^ㄱ-힣a-zA-Z0-9]+", ' ', x).strip().split()
    # RuntimeError: No matching overloads found. at src/native/common/jp_method.cpp:121
    # 에러가 뜨는 경우에는 argument를 잘 확인해 주어야 한다. 현재의 경우 str이 들어가야 하는 자리에 list형식이 들어가있다.
        tmp.append(twitter.pos(str(x), norm=True, stem=True))
        
    #      if twitter.pos(str(x))[1] is not 'Punctuation')
    #     tmp = [s for s in tmp[i] if (s[1] != 'Punctuation' or s[1] != 'Alpha')]
    #     (b.append(a) if a is not None else None)
        for j in tmp[i]:
            set_words.add(j)
        if len(tmp[i]) > max_count:
            max_count = len(tmp[i])

    return tmp, max_count , set_words