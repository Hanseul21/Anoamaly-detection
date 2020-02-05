from Evaluation import evaluate

_, recall, prec, th, f1 = evaluator.pr_auc(y_raw, y)
                # print(e*batch_n + b)
                # print(y_raw)
                # print('recall   :', recall)
                # print('prec     :', prec)
                # print('threshold:', th)
                # print('f1 score :', f1)
                argmax = np.argmax(f1)
                y_pred = evaluator.get_score(y_raw, th[argmax])
                new_pred = evaluator.anomaly_seg(y_pred, y)
                # f1 = evaluator.f1(y_pred, y)
                f1_corr = evaluator.f1(new_pred, y)
                print('Best f1-score {0:.5f} corrected f1-score {1:.5f}'.format(np.max(f1), f1_corr))
                print('recall {0:.5f} precision {1:.5f}'.format(recall[argmax], prec[argmax]))
                print('true', true)
                evaluator.record(e * batch_n + b, recall[argmax], prec[argmax], f1[argmax], true, th[argmax])
                print('recorded')

                recon_prob = evaluator.get_log_prob(recon_x, mu=x_mu, std=x_std)

                _, recall_prob, prec_prob, th_prob, f1_prob = evaluator.pr_auc(recon_prob, y)
                argmax_ = np.argmax(f1_prob)
                recall_prob, prec_prob, th_prob, f1_prob = \
                    recall_prob[argmax_], prec_prob[argmax_], th_prob[argmax_], f1_prob[argmax_]

                print('reconstruction probability')
                print('recall :{0:.4f}, precision :{1:.4f}, threshold :{2:.4f},'
                      ' f1 score :{3:.4f}'.format(recall_prob, prec_prob, th_prob, f1_prob))