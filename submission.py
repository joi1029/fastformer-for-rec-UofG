# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import torch
import numpy as np
import zipfile

from utility.utils import (setuplogger,  dump_args, check_args_environment)
from data_handler.preprocess import get_news_feature, infer_news
from data_handler.TestDataloader import DataLoaderLeader
from models.speedyrec import MLNR

from time import perf_counter
from datetime import timedelta

def generate_submission(args):
    setuplogger()
    args = check_args_environment(args)
    logging.info('-----------start test------------')

    local_rank = 0
    device = torch.device('cuda', int(local_rank))

    model = MLNR(args)
    model = model.to(device)
    ckpt = torch.load(args.load_ckpt_name)
    model.load_state_dict(ckpt['model_state_dict'])

    with open('time.txt', 'a') as file:
        # for i in [5, 10, 15, 20, 24, 'all']:
        for i in [5, 10, 15]:
            start = perf_counter()
            prediction(model, args, device, ckpt['category_dict'], ckpt['subcategory_dict'], n=i)
            end = perf_counter()
            file.write(f"For User History of size {i} took {timedelta(end-start)} minutes.\n")


def reldiff(user, user_history, candidate_news):
    rd = []
    for n in candidate_news:
        cn = n * user_history
        l2 = np.linalg.norm(cn, axis=1)
        l2 = np.stack([norm if norm != 0 else 1 for norm in l2])
        rd.append(user - (cn.T / l2).T)
    return np.mean(rd, axis=1)


def prediction(model, args, device, category_dict, subcategory_dict, n):
    model.eval()
    with torch.no_grad():
        news_info, news_combined = get_news_feature(args, mode='test', category_dict=category_dict,
                                                    subcategory_dict=subcategory_dict)
        news_vecs = infer_news(model, device, news_combined)

        dataloader = DataLoaderLeader(
            news_index=news_info.news_index,
            news_scoring=news_vecs,
            data_dirs=os.path.join(args.root_data_dir,
                                    f'test/'),
            filename_pat=args.filename_pat,
            args=args,
            world_size=1,
            worker_rank=0,
            cuda_device_idx=0,
            enable_prefetch=False,
            enable_shuffle=args.enable_shuffle,
            enable_gpu=args.enable_gpu,
        )

        f = open('prediction.txt', 'w', encoding='utf-8')


        # for cnt, (impids, log_vecs, log_mask, candidate_vec) in enumerate(dataloader.generate_batch()):
        for cnt, (impids, log_vecs, log_mask, candidate_vec, user_news_history) in enumerate(dataloader.generate_batch()):  # ADDED

            if args.enable_gpu:
                log_vecs = log_vecs.cuda(device=device, non_blocking=True)
                log_mask = log_mask.cuda(device=device, non_blocking=True)

            user_vecs = model.user_encoder(
                log_vecs, log_mask, user_log_mask=True).to(torch.device("cpu")).detach().numpy()

            # for id, user_vec, news_vec in zip(
            #         impids, user_vecs, candidate_vec):
            for id, user_vec, news_vec, hist_vec in zip(  # ADDED
                    impids, user_vecs, candidate_vec, user_news_history):  # ADDED
                
#                 score = np.dot(
#                     news_vec, user_vec
#                 )
                
#                 print(news_vec.shape, user_vec.shape, hist_vec.numpy(force=True).shape)  # ADDED
                 #      (21, 256)       (256,)          torch.Size([100, 256])
#                 score_reldiff = score  # ADDED
                if n != 'all':
                    size = min(n, len(hist_vec))
                else:
                    size = len(hist_vec)

                score_reldiff = [np.dot(new, usr) for new, usr in zip(
                   news_vec,
                   reldiff(user_vec, hist_vec[:size], news_vec)
                )]
                # score_reldiff = np.dot(reldiff(user_vec, hist_vec[:size], news_vec), user_vec)

                # pred_rank = (np.argsort(np.argsort(score)[::-1]) + 1).tolist()
                pred_rank = (np.argsort(np.argsort(score_reldiff)[::-1]) + 1).tolist()  # ADDED
                f.write(str(id) + ' ' + '[' + ','.join([str(x) for x in pred_rank]) + ']' + '\n')
                if id in [11, 911]:
                    print(f"\nu{id}", pred_rank)

        f.close()

    zip_file = zipfile.ZipFile(f'ff-base-v5.0-{n}.zip', 'w', zipfile.ZIP_DEFLATED)
    zip_file.write('prediction.txt')
    zip_file.close()
    os.remove('prediction.txt')

if __name__ == "__main__":
    from parameters import parse_args
    setuplogger()
    args = parse_args()
    generate_submission(args)
