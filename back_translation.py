from translate import translate
import pickle
import time

def read_key():
    keys = set()
    for elem in open('back_translation_result.txt'):
        try:
            id, role, _, _, _ = elem.strip().split('\t')
            keys.add((id, role))
        except:
            print(id, role)
    return keys

if __name__ == '__main__':
    filename = 'data_query.pk'
    results = pickle.load(open(filename, 'rb'))

    fileout = open('back_translation_result.txt', 'a')

    while True:
        keys = read_key()
        try:
            idx = 0
            for elem in results:
                idx += 1

                print(idx)

                id, role, query = elem
                if (id, role) in keys:
                    continue

                time.sleep(1)
                chinese = translate(query, 'en', 'zh')
                time.sleep(1)
                english = translate(chinese, 'zh', 'en')

                print('\t'.join([id, role, query, chinese, english]), file=fileout)
        except:
            pass


# ps aux | grep python