import pandas
import math

test_df_size = 1206


def get_correct(acc: float) -> int:
    return math.floor(acc * test_df_size)


def get_incorrect(acc: float) -> int:
    return test_df_size - get_correct(acc)


def get_num_predictions_different(predictions: list[str], test_predictions: list[str]) -> int:
    num_diff_lines = 0
    for index, (pred, test_pred) in enumerate(zip(predictions, test_predictions)):
        if pred != test_pred:
            num_diff_lines += 1

    return num_diff_lines


def get_result(num_predictions_different: int, num_correct: int, num_incorrect: int):
    max_correct = min(test_df_size + (num_incorrect - num_predictions_different), test_df_size)
    min_correct = max(test_df_size - (num_incorrect + num_predictions_different), 0)

    print(f'Number of different predictions: {num_diff_predictions93}')
    print(f'Max corrects: {max_correct}')
    print(f'Min corrects: {min_correct}')

    print(f'{min_correct / test_df_size} <= acc <= {max_correct / test_df_size}')


submission93 = open('submission_093074.csv').readlines()  # this submission has an accuracy of 0.93074
submission27 = open('submission_027977.csv').readlines()  # this submission has an accuracy of 0.27977
submission = open('submission_v2.csv').readlines()

num_diff_predictions93 = get_num_predictions_different(submission93, submission)
num_diff_predictions27 = get_num_predictions_different(submission27, submission)

submission93_correct = get_correct(0.93074)
submission93_incorrect = get_incorrect(0.93074)

get_result(num_diff_predictions93, submission93_correct, submission93_incorrect)

submission27_correct = get_correct(0.27977)
submission27_incorrect = get_incorrect(0.27977)

# get_result(num_diff_predictions27, submission27_correct, submission27_incorrect)
# %%
