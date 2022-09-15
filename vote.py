import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

import warnings
warnings.filterwarnings("ignore")

import argparse
import jsonlines

from tqdm import tqdm
from collections import Counter


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file_name_ls", action="append", required=True, help="读取文件的列表")
    parser.add_argument("--save_path",                     required=True, help="最后结果的保存位置")

    args = parser.parse_args()

    print("[1] 一共融合 {} 个文件, 使用投票的方式进行表决。".format(len(args.file_name_ls)))

    assert len(args.file_name_ls) % 2 != 0, "投票方式表决需要奇数个文件！"

    # ---------------------- #
    # 1. 创建文件对齐
    # ---------------------- #
    all_file_ls = [[] for _ in range(len(args.file_name_ls))]
    for i, file_path in enumerate(tqdm(args.file_name_ls, desc="[2] 正在读取文件")):
        results_ls = jsonlines.open(file_path)
        for result in results_ls:
            all_file_ls[i].append(result)
    
    all_file_ls = list(zip(*all_file_ls))

    # ---------------------- #
    # 2. 融合各个文件判断
    # ---------------------- #
    result_before_voting = list()

    for prediction in tqdm(all_file_ls, desc="[3] 正在进行融合"):
        assert len(set(list(map(lambda x: x["id"], prediction)))) == 1, "[×] 文件对齐错误！"

        result_dict_to_add, temp_dict = dict(), dict()
        result_dict_to_add["id"] = prediction[0]["id"]

        for pred in prediction:
            for token_pred in pred["predictions"]:
                if token_pred["id"] not in temp_dict.keys():
                    temp_dict[token_pred["id"]] = [token_pred["type_id"]]
                else:
                    temp_dict[token_pred["id"]].append(token_pred["type_id"])
        
        result_dict_to_add["predictions"] = temp_dict
        result_before_voting.append(result_dict_to_add)

    # ---------------------- #
    # 3. 整理成要提交的格式
    # ---------------------- #
    result_to_vote, confused_num, cnt = list(), 0, 0

    for result in tqdm(result_before_voting, desc="[3] 整理成要提交的格式"):
        sentence_pred = { "id": result["id"], "predictions": list() }
        for key in result["predictions"].keys():
            n_pred_counter = Counter(result["predictions"][key])
            votted_winner = n_pred_counter.most_common(1)[0]

            if votted_winner[1] < len(args.file_name_ls) / 2.0:
                confused_num += 1
                if votted_winner[1] == 1:
                    cnt += 1
            
            sentence_pred["predictions"].append({ "id": key, "type_id": votted_winner[0] })

        result_to_vote.append(sentence_pred)

    print("[√] 表决力度不够大的样本个数有 {} 个".format(confused_num))
    print("[√] 所有模型表决不一的样本个数 {} 个".format(cnt))

    # ---------------------- #
    # 4. 写入文件
    # ---------------------- #
    with jsonlines.open(args.save_path, mode='w') as writer:
        for line in result_to_vote:
            writer.write(line)


if __name__ == "__main__":
    main()