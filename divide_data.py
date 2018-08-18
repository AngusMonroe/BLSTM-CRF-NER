# -*- coding: utf-8 -*-


def divide_data(lower=True):
    data_file = open("dataset/aminer/data.txt", "r", encoding="utf8")
    train_file = open("dataset/aminer/aminer_train.dat", "w", encoding="utf8")
    dev_file = open("dataset/aminer/aminer_dev.dat", "w", encoding="utf8")
    test_file = open("dataset/aminer/aminer_test.dat", "w", encoding="utf8")
    train_test_file = open("dataset/aminer/aminer_train_test.dat", "w", encoding="utf8")
    lines = data_file.readlines()
    num = len(lines)
    print(num)

    file = train_file
    print('Writing in train_file...')
    for i, line in enumerate(lines):
        if lower:
            line = line.lower()
        # train: dev: test: train_test = 80: 8: 8: 4
        file.write(line)
        if line == '\n':
            if file == train_file and i > num * 0.8:
                file.write('-DOCSTART- -X- O O')
                file = dev_file
                print('Writing in dev_file...')
            elif file == dev_file and i > num * 0.88:
                file.write('-DOCSTART- -X- O O')
                file = test_file
                print('Writing in test_file...')
            elif file == test_file and i > num * 0.96:
                file.write('-DOCSTART- -X- O O')
                file = train_test_file
                print('Writing in train_test_file...')

    data_file.close()
    train_file.close()
    dev_file.close()
    test_file.close()
    train_file.close()

    print("done")

def transfer_to_lower():
    file_path = ["aminer_train.dat",
                 "aminer_dev.dat",
                 "aminer_test.dat",
                 "aminer_train_test.dat"]
    for path in file_path:
        file = open("dataset/aminer_segment_bak/" + path, "r", encoding="utf8")
        out_file = open("dataset/aminer_segment/" + path, "w", encoding="utf8")
        lines = file.readlines()
        for line in lines:
            if line != lines[-1]:
                word = line.split()
                if len(word) > 1:
                    word[0] = word[0].lower()
                    line = word[0] + ' ' + word[1] + '\n'
            out_file.write(line)
        file.close()
        out_file.close()
    print('done')


if __name__ == '__main__':
    divide_data(lower=True)
    # transfer_to_lower()
