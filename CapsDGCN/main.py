from tqdm import tqdm
import os, sys
import logging
import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from dataloader import DialogLoader
from model import DialogBertTransformer
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report,multilabel_confusion_matrix,jaccard_score, hamming_loss, precision_score, recall_score,zero_one_loss


from utils import Processor, set_seed

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    "Prepare optimizer"
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return optimizer
    
def configure_dataloaders(dataset, batch_size):
    "Prepare dataloaders"
        
    train_loader = DialogLoader(
        'datasets/' + dataset + '/' + 'train.csv',
        'datasets/' + dataset + '/' + 'train.stanford.json',
        batch_size,
        dataset,
        shuffle=True
    )
    
    valid_loader = DialogLoader(
        'datasets/' + dataset + '/' + 'valid.csv',
        'datasets/' + dataset + '/' + 'valid.stanford.json',
        batch_size,
        dataset,
        shuffle=True
    )
    
    test_loader = DialogLoader(
        'datasets/' + dataset + '/' + 'test.csv',
        'datasets/' + dataset + '/' + 'test.stanford.json',
        batch_size,
        dataset,
        shuffle=True
    )
    
    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, mode, loss_function_pol, loss_function_emo, dataloader, epoch, args , loss_weights_pol, loss_weights_emo, optimizer=None, train=False, one_element=False):
    losses, pol_preds, pol_labels, emo_preds, emo_labels, masks = [], [], [], [], [], []
    assert not train or optimizer!=None

    if train:
        model.train()
    else:
        model.eval()
    
    for conversations, pol_label, emo_label, deptree in tqdm(dataloader, leave=False):
    
        processor = Processor(direct=True) 
        
        if train:
            optimizer.zero_grad()
        
        # create labels
        pol_label = torch.nn.utils.rnn.pad_sequence([torch.tensor(pol_label)],
                                                batch_first=True).to(device)
        if dataset == 'poem':                                        
            emo_label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in emo_label],
                                                batch_first=True).to(device)
        elif  dataset == 'dailydialog':   
            emo_label = torch.nn.utils.rnn.pad_sequence([torch.tensor(emo_label)],
                                                batch_first=True).to(device)
        # obtain log probabilities
        log_prob_pol, log_prob_emo = model(conversations, processor, mode, deptree)
        
        # compute loss and metrics
        if dataset == 'poem':
            pol_labels_ = pol_label.view(-1).to(device)
            pol_lp_ = log_prob_pol.view(-1,log_prob_pol.size()[1])
            
            emo_labels_ = emo_label.view(-1,emo_label.size()[1]).type(torch.FloatTensor).to(device)
            emo_lp_ = log_prob_emo.view(-1, log_prob_emo.size()[1])
            
        elif dataset == 'dailydialog':
            pol_labels_ = pol_label.view(-1).to(device)
            pol_lp_ = log_prob_pol.view(-1,log_prob_pol.size()[1])
            
            emo_labels_ = emo_label.view(-1).to(device)
            emo_lp_ = log_prob_emo.view(-1, log_prob_emo.size()[1])

        loss_pol = loss_function_pol(pol_lp_, pol_labels_)
        loss_emo = loss_function_emo(emo_lp_, emo_labels_)

        pol_pred_ = torch.argmax(pol_lp_, 1)
        
        pol_preds.append(pol_pred_.data.cpu().numpy())
        pol_labels.append(pol_labels_.data.cpu().numpy())
        
        
        if dataset == 'poem':    
            emo_pred_ = ((torch.sigmoid(emo_lp_)) > 0.5).int()    
        elif dataset == 'dailydialog':
            emo_pred_ = torch.argmax(emo_lp_, 1)
        
        emo_preds.append(emo_pred_.data.cpu().detach().numpy())
        
        if dataset == 'poem':
            emo_labels.append(emo_labels_.int().data.cpu().detach().numpy())
        elif dataset == 'dailydialog':
            emo_labels.append(emo_labels_.data.cpu().detach().numpy())
        
        # save_grad = True
        loss = loss_pol + loss_emo
        
        losses.append(loss.item())
        
        if train:
            loss.backward()
            optimizer.step()
           
    if pol_preds!=[] and emo_preds!=[]:
        pol_preds  = np.concatenate(pol_preds)
        pol_labels = np.concatenate(pol_labels)
        emo_preds  = np.concatenate(emo_preds)
        emo_labels  = np.concatenate(emo_labels)
        
        
    else:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), [], [], [], []
    
    avg_loss = round(np.mean(losses), 4)
    avg_accuracy_pol = round(accuracy_score(pol_labels, pol_preds)*100, 2)
    avg_accuracy_emo = round(accuracy_score(emo_labels, emo_preds)*100, 2)


    if dataset in ['poem','dailydialog']:
        avg_fscore_pol = round(f1_score(pol_labels, pol_preds, average='weighted', zero_division=0)*100, 2)
        avg_fscore_emo = round(f1_score(emo_labels, emo_preds, average='weighted', zero_division=0)*100, 2)
        fscores_pol = [avg_fscore_pol]
        fscores_emo = [avg_fscore_emo]
        if one_element:
            fscores_pol = fscores_pol[0]
            fscores_emo = fscores_emo[0]
    
    return avg_loss, avg_accuracy_pol, avg_accuracy_emo, fscores_pol, fscores_emo, pol_labels, emo_labels, pol_preds, emo_preds
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--cls-model', default='lstm', help='lstm or logreg')
    parser.add_argument('--model', default='roberta', help='which model family bert|roberta|sbert; sbert is sentence transformers')
    parser.add_argument('--mode', default='0', help='which mode 0: bert or roberta base | 1: bert or roberta large; \
                                                     0, 1: bert base, large sentence transformer and 2, 3: roberta base, large sentence transformer')
    parser.add_argument('--max-seq-len', default=180, help='maximum sequence length; 300 for dailydialog and 180 for podial')
    parser.add_argument('--dataset', default='poem', help='which dataset')
    parser.add_argument('--gcn_layer_number', type=int, default=1)
    parser.add_argument('--run' , help='which run')
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")

    parser.add_argument('--inference', default=None, help='model ID')
    
    parser.add_argument('--gpu-no', type=int, default=3, metavar='GN', help='GPU Number')

    args = parser.parse_args()

    print(args)
    
    set_seed(args)
    
    gpu_no = args.gpu_no
    
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:'+str(gpu_no) if torch.cuda.is_available() else 'cpu')
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    
    
    if not args.inference:
        run_ID = int(time.time())

    global dataset
    dataset = args.dataset
    
    batch_size = args.batch_size
    n_epochs = args.epochs
    classification_model = args.cls_model
    transformer_model = args.model
    transformer_mode = args.mode
    
    
    if dataset == 'poem':
        print ('Classifying in poem.')
        n_classes_pol  = 3
        n_classes_emo = 17
        loss_weights_pol = torch.FloatTensor([0.93272435, 2.87276621, 0.6330014 ])
        loss_weights_emo = torch.FloatTensor([0.16692769455874845, 0.33005145521105966, 0.3880637213338786, 0.3761669704456123, 0.5639260139738498, 0.5886589748439569, 0.6815820594805309, 0.7171456486072962, 0.923118904681679, 0.9760766750718202, 1.8395592963873653, 1.9070098039215686, 3.603798054654933, 3.8918567426970787, 5.990606713889744, 9.535049019607843, 23.714111551356293])
        

    elif dataset == 'dailydialog':
        print ('Classifying in dailydialog.')
        n_classes_pol  = 3
        n_classes_emo = 7
        loss_weights_pol = torch.FloatTensor([ 1.44087408, 19.24282561,  0.44365387])
        loss_weights_emo = torch.FloatTensor([ 0.17261352, 15.05786837, 41.09853843, 85.29354207,  1.11365204, 12.85124576, 7.78303571])

    else:
        raise ValueError('--dataset must be poem or dailydialog')

    train_loader, valid_loader, test_loader = configure_dataloaders(dataset, batch_size)
    


    model = DialogBertTransformer(args,D_h, classification_model, transformer_model, transformer_mode, n_classes_pol, n_classes_emo, context_attention, device, attention, residual)

    model = model.to(device)
    
    if args.inference:
        if dataset == 'poem':
            model.load_state_dict(torch.load(f'saved/poem/{args.inference}.pt'))

        elif dataset == 'dailydialog':
            model.load_state_dict(torch.load(f'saved/dailydialog/{args.inference}.pt'))

        n_epochs = 1

   
    if args.class_weight:
        loss_function_pol  = nn.CrossEntropyLoss(loss_weights_pol.to(device))
        if dataset == 'poem':
            loss_function_emo  = nn.BCEWithLogitsLoss(weight=loss_weights_emo.to(device))
        else:
            loss_function_emo  = nn.CrossEntropyLoss(loss_weights_emo.to(device))
    else:
        loss_function_pol  = nn.CrossEntropyLoss()
        if dataset == 'poem':
            loss_function_emo  = nn.BCEWithLogitsLoss()
        else:
            loss_function_emo  = nn.CrossEntropyLoss()
            

    if not args.inference:
        optimizer = configure_optimizers(model, args.weight_decay, args.lr, args.adam_epsilon)

    valid_losses, valid_fscores_pol, valid_fscores_emo = [], [], []
    test_fscores_pol, test_fscores_emo = [], []
    best_loss, best_label_pol, best_pred_pol, best_fscore_pol, best_label_emo, best_pred_emo, best_fscore_emo = None, None, None, None, None, None, None
    
    last_loss = float('inf')
    patience = 4
    trigger_times = 0


    for e in tqdm(range(n_epochs)):
        start_time = time.time()
        
        if not args.inference:
            train_loss, train_acc_pol, train_acc_emo, train_fscore_pol, train_fscore_emo, _, _, _, _ = train_or_eval_model(model, 'train', loss_function_pol, loss_function_emo,
                                                                           train_loader, e, args, loss_weights_pol, loss_weights_emo,optimizer=optimizer if not args.inference else None,
                                                                           train=True if not args.inference else False,
                                                                           one_element=True)
                                                                           


            valid_loss, valid_acc_pol, valid_acc_emo, valid_fscore_pol, valid_fscore_emo, _, _, _, _ = train_or_eval_model(model, 'valid',                                                              loss_function_pol, loss_function_emo,
                                                                           valid_loader, e, args, loss_weights_pol, loss_weights_emo,one_element=True
                                                                           )


            valid_losses.append(valid_loss)
            valid_fscores_pol.append(valid_fscore_pol)
            valid_fscores_emo.append(valid_fscore_emo)
        
        test_loss, test_acc_pol, test_acc_emo, test_fscore_pol, test_fscore_emo, test_label_pol, test_label_emo, test_pred_pol, test_pred_emo  = train_or_eval_model(model, 'test', loss_function_pol, loss_function_emo,
                        test_loader, e, args, loss_weights_pol, loss_weights_emo, one_element=True)
        test_fscores_pol.append(test_fscore_pol)
        test_fscores_emo.append(test_fscore_emo)

        # WARNING: model hyper-parameters are not stored
        if not args.inference:
            #Early Stopping
            
            """
            if valid_loss >= last_loss:
                trigger_times += 1
                
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break
                    
            last_loss = valid_loss
            """
            if best_loss == None or valid_loss < best_loss:
                best_loss = valid_loss
                best_fscore_pol = valid_fscore_pol
                best_fscore_emo = valid_fscore_emo
                if not os.path.exists('mapping/'):
                    os.makedirs('mapping/')
                with open(f'mapping/{dataset}_run{args.run}_{run_ID}.tsv', 'w') as f:
                    f.write(f'{args}\tRun ID: {run_ID}\t\t Best Loss: {best_loss}\t Best F1 Polite: {best_fscore_pol}\tBest F1 Emo: {best_fscore_emo}')
                
                if dataset == 'poem':
                    dirName = 'saved/poem/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/poem/run{args.run}_{run_ID}_{args.model}_{args.cls_model}.pt')

                elif dataset == 'dailydialog':
                    dirName = 'saved/dailydialog/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/dailydialog/run{args.run}_{run_ID}_{args.model}_{args.cls_model}.pt')

        if not args.inference:
            if best_loss == None or best_loss > valid_loss:
                best_loss, best_label_pol, best_label_emo, best_pred_pol, best_pred_emo, =\
                    valid_loss, test_label_pol, test_label_emo, test_pred_pol, test_pred_emo
            x = 'Epoch {} train_loss {} train_acc_pol {} train_acc_emo {} train_fscore_pol {} train_fscore_emo {} valid_loss {} valid_acc_pol {} valid_acc_emo {} valid_fscore_pol {} valid_fscore_emo {} test_loss {} test_acc_pol {} test_acc_emo {} test_fscore_pol {} test_fscore_emo {} time {}'.\
                format(e+1, train_loss, train_acc_pol, train_acc_emo, train_fscore_pol, train_fscore_emo, valid_loss, valid_acc_pol, valid_acc_emo, valid_fscore_pol,\
                        valid_fscore_emo, test_loss, test_acc_pol, test_acc_emo, test_fscore_pol, test_fscore_emo, round(time.time()-start_time, 2))
                        
            print (x)

    test_fscores_pol = np.array(test_fscores_pol).transpose()
    test_fscores_emo = np.array(test_fscores_emo).transpose()
    
    if not args.inference:
        sys.exit(0)
        
    else:
        
        best_label_pol, best_label_emo, best_pred_pol, best_pred_emo = test_label_pol, test_label_emo, test_pred_pol, test_pred_emo
        
        if not os.path.exists('results/'):
            os.makedirs('results/')
        lf = open('results/' + dataset + '_' + transformer_model + '_' + classification_model + '.txt', 'a')
        
        lf.write(str(args.inference) + '\t' + str(args) + '\n')
        lf.write('-'*50 + '\n\n')
        lf.write('\n Politeness Classification Metrics\n')
        lf.write('\n Accuracy: ' + str(accuracy_score(best_label_pol, best_pred_pol)) + '\n')
        lf.write('\n Macro Precision: ' + str(precision_score(best_label_pol, best_pred_pol, average='macro', zero_division=0)) + '\n')
        lf.write('\n Macro Recall: ' + str(recall_score(best_label_pol, best_pred_pol, average='macro', zero_division=0)) + '\n')
        lf.write('\n Macro F1-score: ' + str(f1_score(best_label_pol, best_pred_pol, average='macro',zero_division=0)) + '\n')
        lf.write('-'*50 + '\n\n')
        
        lf.write('\n Emotion Classification Metrics\n')
        
        if args.dataset == 'poem':
            lf.write('\n Exact Match Ratio/Subset Accuracy/0/1 Accuracy: '+str(round(accuracy_score(best_label_emo, best_pred_emo)*100, 2)) + '\n')
            lf.write('\n Micro Precision: ' + str(precision_score(best_label_emo, best_pred_emo, average='micro', zero_division=0)) + '\n')
            lf.write('\n Micro Recall: ' + str(recall_score(best_label_emo, best_pred_emo, average='micro', zero_division=0)) + '\n')
            lf.write('\n Micro F1-score: ' + str(f1_score(best_label_emo, best_pred_emo, average='micro', zero_division=0)) + '\n')
            lf.write('\n Hamming Loss: ' + str(hamming_loss(best_label_emo, best_pred_emo)) + '\n')
            lf.write('\n Weighted Jaccard Index: ' + str(jaccard_score(best_label_emo, best_pred_emo,average='weighted')) + '\n')
            lf.write('\n 0/1 Loss: ' + str(zero_one_loss(best_label_emo, best_pred_emo)) + '\n')
        else:
            lf.write('\n Accuracy: ' + str(accuracy_score(best_label_emo, best_pred_emo)) + '\n')
            lf.write('\n Macro Precision: ' + str(precision_score(best_label_emo, best_pred_emo, average='macro', zero_division=0)) + '\n')
            lf.write('\n Macro Recall: ' + str(recall_score(best_label_emo, best_pred_emo, average='macro', zero_division=0)) + '\n')
            lf.write('\n Macro F1-score: ' + str(f1_score(best_label_emo, best_pred_emo, average='macro',zero_division=0)) + '\n')
        lf.close()
        print('Done')

