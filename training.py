# 用于执行单轮模型训练和学习率更新
import torch
from tqdm import tqdm

#定义一轮
def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
):
    model.train()

    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 根据模型输出计算损失
        loss = criterion(
            outputs.logits,
            labels
        )
        #计算梯度
        loss.backward()

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    return average_loss
