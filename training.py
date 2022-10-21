import torch
import wandb
import tqdm
import gc

def train(model, train_loader, validation_loader, epochs, optimizer, evaluation_per_step=10, acc_step=1):
    wandb.watch(model, log_freq=evaluation_per_step)
    for epoch in range(epochs):
        print(f'+++++++++++++++++++++++++++++++++++++++++ epoch: {epoch + 1} ++++++++++++++++++++++++++++++++++++++++++++')
        print('training...')

        cnt = 0
        total_loss = 0

        model.train()

        for i, (element1, element2) in enumerate(train_loader):
            print(f'--------------------------- train loader {i} ----------------------------------')
            cnt += 1
            # to cuda?
            # element1 = element1.to('cuda')
            # element2 = element2.to('cuda')
            if cnt % evaluation_per_step == 0:
                accuracy1 = evaluate_model(model, validation_loader)
                accuracy2 = evaluate_model(model, train_loader)

                accuracy1 = accuracy1.view(-1).cpu().item()
                accuracy2 = accuracy2.view(-1).cpu().item()
                print(f'count: {cnt}')
                print(f'********accuracy: {accuracy1} ********')
                print(f'********loss: {total_loss / evaluation_per_step} ********')
                
                wandb.log({"train/train-acc": accuracy2, "train/eval-acc": accuracy1, "train/loss": total_loss / evaluation_per_step})

                total_loss = 0
            # if i % 20 == 0:
            #     torch.cuda.empty_cache()
            #     print('torch cache cleaned!')
            # torch.cuda.empty_cache()
            # print('torch cache cleaned!')
            
            if cnt % acc_step == 0:
                loss = model.get_loss(model((element1, element2)))
                total_loss += loss.detach().cpu().item()
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                print('torch cache cleaned!')
                print(f'start backward {cnt}...')
                loss.backward()
                print(f'successfully end backward {cnt}')
                optimizer.step()
            else:
                loss = model.get_loss(model((element1, element2)))
                loss.backward()

        torch.save(model.state_dict(), './autodl-tmp/saved_model/' + str(epoch + 100) + '.pth')
        print(f'saved epoch {epoch} model successfully!')


def evaluate_model(model, loader):
    print('##########evaluating#################')
    model.eval()
    with torch.no_grad():
        # correct = torch.tensor([0]).cuda()
        # total = torch.tensor([0]).cuda()
        correct = 0
        total = 0
        for i, (element1, element2) in enumerate(tqdm.tqdm(loader)):
            right, num = model.get_accuracy(model((element1, element2)))
            right.to(torch.device('cpu'))
            correct += right
            total += num
        print('##########evaluate successful!########')
    return correct / total
