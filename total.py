def build_dataset(train_text):
    words = list()
    with open(train_text, 'r') as f:
        lines = f.readlines()
        for line in lines:
            #  line -> line.split('\t')[1]
            sentence = line.split('\t')[1]
            if sentence:
                words.append(sentence)

    with open('./total_naver.txt','w') as f:
        for lines in words:
            f.write(lines+'\n')

if __name__ == '__main__':
    build_dataset('네이버_크롤링.txt')