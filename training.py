import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from contrax_loss import compute_sim_matrix, compute_target_matrix, contrastive_loss
from dataset import BertDataset
from dataset import TrainSamplerMultiClassUnit
from models import BertClassifier
from models import LogisticRegression
from utils import *  # bad practice, nvm

ckpt_dir = 'exp_data'


def train_bert(train_dict, test_dic, tqdm_on, model_name, embed_len, id, num_epochs, base_bs, base_lr,
               mask_classes, coefficient, num_authors, val_dic=None):
    print(f'mask classes = {mask_classes}')

    # tokenizer and pretrained model
    tokenizer, extractor = None, None
    if 'bert-base' in model_name:
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(model_name)
        extractor = BertModel.from_pretrained(model_name)
    elif 'deberta' in model_name:
        from transformers import DebertaTokenizer, DebertaModel
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        extractor = DebertaModel.from_pretrained(model_name)
    else:
        raise NotImplementedError(f"model {model_name} not implemented")

    # update extractor
    for param in extractor.parameters():
        param.requires_grad = True

    # get dataset
    train_x, train_y = train_dict['content'].tolist(), train_dict['Target'].tolist()
    test_x, test_y = test_dic['content'].tolist(), test_dic['Target'].tolist()

    if val_dic is not None:
        val_x, val_y = val_dic['content'].tolist(), val_dic['Target'].tolist()

    # training config
    ngpus, dropout = torch.cuda.device_count(), 0.35
    num_tokens, hidden_dim, out_dim = 256, 512, num_authors
    model = BertClassifier(extractor, LogisticRegression(embed_len * num_tokens, hidden_dim, out_dim, dropout=dropout))
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=base_lr * ngpus, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_set = BertDataset(train_x, train_y, tokenizer, num_tokens)
    test_set = BertDataset(test_x, test_y, tokenizer, num_tokens)

    if val_dic is not None:
        val_set = BertDataset(val_x, val_y, tokenizer, num_tokens)

    temperature, sample_unit_size = 0.1, 2
    print(f'coefficient, temperature, sample_unit_size = {coefficient, temperature, sample_unit_size}')

    # logger
    exp_dir = os.path.join(ckpt_dir,
                           f'{id}_{model_name.split("/")[-1]}_coe{coefficient}_temp{temperature}_unit{sample_unit_size}_epoch{num_epochs}')
    writer = SummaryWriter(os.path.join(exp_dir, 'board'))

    # load data
    train_sampler = TrainSamplerMultiClassUnit(train_set, sample_unit_size=sample_unit_size)
    train_loader = DataLoader(train_set, batch_size=base_bs * ngpus, sampler=train_sampler, shuffle=False,
                              num_workers=4 * ngpus, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                             pin_memory=True, drop_last=True)

    if val_dic is not None:
        val_loader = DataLoader(val_set, batch_size=base_bs * ngpus, shuffle=False, num_workers=4 * ngpus,
                                pin_memory=True, drop_last=True)

    final_test_acc = None
    final_train_preds, final_test_preds = [], []
    best_acc = -1
    best_tv_acc = -1

    # training loop
    for epoch in range(num_epochs):
        train_acc = AverageMeter()
        train_loss = AverageMeter()
        train_loss_1 = AverageMeter()
        train_loss_2 = AverageMeter()

        # decay coefficient
        # coefficient = coefficient - 1 / num_epochs

        # training
        model.train()
        pg = tqdm(train_loader, leave=False, total=len(train_loader), disable=not tqdm_on)
        for i, (x1, x2, x3, y) in enumerate(pg):  # for x1, x2, x3, y in train_set:
            x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
            pred, feats = model(x, return_feat=True)

            # classification loss
            loss_1 = criterion(pred, y.long())

            # generate the mask
            mask = y.clone().cpu().apply_(lambda x: x not in mask_classes).type(torch.bool).cuda()
            feats, pred, y = feats[mask], pred[mask], y[mask]
            if len(y) == 0:
                continue

            # contrastive learning
            sim_matrix = compute_sim_matrix(feats)
            target_matrix = compute_target_matrix(y)
            loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

            # total loss
            loss = loss_1 + coefficient * loss_2

            acc = (pred.argmax(1) == y).sum().item() / len(y)
            train_acc.update(acc)
            train_loss.update(loss.item())
            train_loss_1.update(loss_1.item())
            train_loss_2.update(loss_2.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pg.set_postfix({
                'train acc': '{:.6f}'.format(train_acc.avg),
                'train L1': '{:.6f}'.format(train_loss_1.avg),
                'train L2': '{:.6f}'.format(train_loss_2.avg),
                'train L': '{:.6f}'.format(train_loss.avg),
                'epoch': '{:03d}'.format(epoch)
            })

            # iteration logger
            step = i + epoch * len(pg)
            writer.add_scalar("train-iteration/L1", loss_1.item(), step)
            writer.add_scalar("train-iteration/L2", loss_2.item(), step)
            writer.add_scalar("train-iteration/L", loss.item(), step)
            writer.add_scalar("train-iteration/acc", acc, step)

        print('train acc: {:.6f}'.format(train_acc.avg), 'train L1 {:.6f}'.format(train_loss_1.avg),
              'train L2 {:.6f}'.format(train_loss_2.avg), 'train L {:.6f}'.format(train_loss.avg), f'epoch {epoch}')

        # epoch logger
        writer.add_scalar("train/L1", train_loss_1.avg, epoch)
        writer.add_scalar("train/L2", train_loss_2.avg, epoch)
        writer.add_scalar("train/L", train_loss.avg, epoch)
        writer.add_scalar("train/acc", train_acc.avg, epoch)

        # validation
        if val_dic is not None:
            model.eval()
            pg = tqdm(val_loader, leave=False, total=len(val_loader), disable=not tqdm_on)
            with torch.no_grad():
                tv_acc = AverageMeter()  # tv stands for train_val
                tv_loss_1 = AverageMeter()
                tv_loss_2 = AverageMeter()
                tv_loss = AverageMeter()
                for i, (x1, x2, x3, y) in enumerate(pg):
                    x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
                    pred, feats = model(x, return_feat=True)

                    # classification
                    loss_1 = criterion(pred, y.long())

                    # contrastive learning
                    sim_matrix = compute_sim_matrix(feats)
                    target_matrix = compute_target_matrix(y)
                    loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

                    # total loss
                    loss = loss_1 + coefficient * loss_2

                    # logger
                    tv_acc.update((pred.argmax(1) == y).sum().item() / len(y))
                    # test_acc.update(
                    #     f1_score(y.cpu().detach().numpy(), pred.argmax(1).cpu().detach().numpy(), average='macro'))
                    tv_loss.update(loss.item())
                    tv_loss_1.update(loss_1.item())
                    tv_loss_2.update(loss_2.item())

                    pg.set_postfix({
                        'train_val acc': '{:.6f}'.format(tv_acc.avg),
                        'epoch': '{:03d}'.format(epoch)
                    })

        # testing
        model.eval()
        pg = tqdm(test_loader, leave=False, total=len(test_loader), disable=not tqdm_on)
        with torch.no_grad():
            test_acc = AverageMeter()
            test_loss_1 = AverageMeter()
            test_loss_2 = AverageMeter()
            test_loss = AverageMeter()
            for i, (x1, x2, x3, y) in enumerate(pg):
                x, y = (x1.cuda(), x2.cuda(), x3.cuda()), y.cuda()
                pred, feats = model(x, return_feat=True)

                # classification
                loss_1 = criterion(pred, y.long())

                # contrastive learning
                sim_matrix = compute_sim_matrix(feats)
                target_matrix = compute_target_matrix(y)
                loss_2 = contrastive_loss(sim_matrix, target_matrix, temperature, y)

                # total loss
                loss = loss_1 + coefficient * loss_2

                # logger
                test_acc.update((pred.argmax(1) == y).sum().item() / len(y))
                # test_acc.update(
                #     f1_score(y.cpu().detach().numpy(), pred.argmax(1).cpu().detach().numpy(), average='macro'))
                test_loss.update(loss.item())
                test_loss_1.update(loss_1.item())
                test_loss_2.update(loss_2.item())

                pg.set_postfix({
                    'test acc': '{:.6f}'.format(test_acc.avg),
                    'epoch': '{:03d}'.format(epoch)
                })

        # logging
        if val_dic is not None:
            writer.add_scalar("tv/L1", tv_loss_1.avg, epoch)
            writer.add_scalar("tv/L2", tv_loss_2.avg, epoch)
            writer.add_scalar("tv/L", tv_loss.avg, epoch)
            writer.add_scalar("tv/acc", tv_acc.avg, epoch)

        writer.add_scalar("test/L1", test_loss_1.avg, epoch)
        writer.add_scalar("test/L2", test_loss_2.avg, epoch)
        writer.add_scalar("test/L", test_loss.avg, epoch)
        writer.add_scalar("test/acc", test_acc.avg, epoch)

        # scheduler.step(test_loss.avg)
        scheduler.step()

        print(f'epoch {epoch}, train acc {train_acc.avg}, test acc {test_acc.avg}')

        final_test_acc = test_acc.avg

        # save model
        if test_acc.avg:
            if test_acc.avg >= best_acc:
                cur_models = os.listdir(exp_dir)
                for cur_model in cur_models:
                    if cur_model.endswith(".pt"):
                        os.remove(os.path.join(exp_dir, cur_model))
                save_model(exp_dir, f'{id}_val{final_test_acc:.5f}_e{epoch}.pt', model)
        best_acc = max(best_acc, test_acc.avg)

        if val_dic is not None:
            print(f'epoch {epoch}, train val acc {tv_acc.avg}')
            final_tv_acc = tv_acc.avg
            best_tv_acc = max(best_tv_acc, tv_acc.avg)

    # save checkpoint
    save_model(exp_dir, f'{id}_val{final_test_acc:.5f}_finale{epoch}.pt', model)

    print(
        f'Training complete after {num_epochs} epochs. Final val acc = {final_tv_acc}, '
        f'best val acc = {best_tv_acc}, best test acc = {best_acc}.'
        f'Final test acc {final_test_acc}')

    return final_test_acc, final_train_preds, final_test_preds
