
tsfm = transforms.Compose([
    transforms.Resize([300, 300]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""TRAIN"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 5
LR = 1e-3
BS = 8
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
print_feq = 100

model = SSD()


criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)

train_ds = SSDDateset(all_img_name[:int(len(all_img_name)*0.8)], transform=tsfm)
train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=train_ds.collate_fn)

valid_ds = SSDDateset(all_img_name[int(len(all_img_name)*0.8):int(len(all_img_name)*0.9)], transform=tsfm)
valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=True, collate_fn=valid_ds.collate_fn)

from tqdm import tqdm
import time
#import pdb;
#pdb.set_trace()
#!pip install -Uqq ipdb
#import ipdb
#%pdb on

for epoch in range(1, EPOCH+1):
    model.train()
    train_loss = []
    for step, (img, boxes, labels) in enumerate(train_dl):
        print(len(boxes), len(img))
        time_1 = time.time()
        #img = img.device
#         box = torch.cat(box)
        boxes = [box for box in boxes]
#         label = torch.cat(label)
        labels = [label for label in labels]

        print(img.shape)
        img = img.to(device)

        pred_loc, pred_sco = model(img)



        loss = criterion(pred_loc, pred_sco, boxes, labels)

         # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         losses.update(loss.item(), images.size(0))
        train_loss.append(loss.item())
        if step % print_feq == 0:
            print('epoch:', epoch,
                  '\tstep:', step+1, '/', len(train_dl) + 1,
                  '\ttrain loss:', '{:.4f}'.format(loss.item()),
                  '\ttime:', '{:.4f}'.format((time.time()-time_1)*print_feq), 's')

    model.eval();
    valid_loss = []
    for step, (img, boxes, labels) in enumerate(tqdm(valid_dl)):
        img = img.device
        boxes = [box.device for box in boxes]
        labels = [label.device for label in labels]
        pred_loc, pred_sco = model(img)
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        valid_loss.append(loss.item())

    print('epoch:', epoch, '/', EPOCH+1,
            '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
            '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)))

model.eval();
del train_ds, train_dl, valid_ds, valid_dl

from random import randint

def test(n=5):
    d = []
    for i in range(5):
        origin_img = Image.open('../input/images/Images/' + all_img_name[-1*randint(1,100)]).convert('RGB')
        img = tsfm(origin_img)

        img = img#.cuda()
        predicted_locs, predicted_scores = model(img.unsqueeze(0))
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2,
                                                                 max_overlap=0.5, top_k=200)
        det_boxes = det_boxes[0].to('cpu')

        origin_dims = torch.FloatTensor([origin_img.width, origin_img.height, origin_img.width, origin_img.height]).unsqueeze(0)
        det_boxes = det_boxes * origin_dims

        annotated_image = origin_img
        draw = ImageDraw.Draw(annotated_image)

        box_location = det_boxes[0].tolist()
        draw.rectangle(xy=box_location, outline='red')
        draw.rectangle(xy=list(map(lambda x:x+1, box_location)), outline='red')
        d.append(annotated_image)
    return d

for i in test():
    plt.figure(figsize=(5, 5))
    plt.imshow(i)