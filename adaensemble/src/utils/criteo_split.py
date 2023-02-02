import fire
from helper import count_lines


def criteo_time_split(
    source_file_name, train_ratio,
    train_file_name, test_file_name
):

    header = "Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26"

    total_lines = count_lines(source_file_name)
    train_lines = int(total_lines * train_ratio)

    source_file = open(source_file_name, "r")
    train_file = open(train_file_name, "w")
    test_file = open(test_file_name, "w")

    idx = 0
    train_file.write(header + "\n")
    test_file.write(header + "\n")

    for line in source_file:
        csvline = line.replace("\t", ",")
        if idx <= train_lines:
            train_file.write(csvline)
        else:
            test_file.write(csvline)
        idx += 1

    source_file.close()
    train_file.close()
    test_file.close()


def criteo_random_split(
    source_file_name, train_ratio,
    train_file_name, test_file_name
):
    header = "Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26"

    source_file = open(source_file_name, "r")
    train_file = open(train_file_name, "w")
    test_file = open(test_file_name, "w")

    idx = 0
    train_file.write(header + "\n")
    test_file.write(header + "\n")

    for line in source_file:
        csvline = line.replace("\t", ",")
        if float(idx) % 1000.0 / 1000.0 <= train_ratio:
            train_file.write(csvline)
        else:
            test_file.write(csvline)
        idx += 1

    source_file.close()
    train_file.close()
    test_file.close()

if __name__ == "__main__":
    fire.Fire({
        "criteo_time_split": criteo_time_split,
        "criteo_random_split": criteo_random_split
    })
