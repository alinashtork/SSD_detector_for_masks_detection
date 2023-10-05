
class SSDDateset(Dataset):
    def __init__(self, file_folder, is_test=False, transform=None):
        self.img_folder_path = '/content/mask_dataset/images'
        self.annotation_folder_path = '/content/mask_dataset/annotations'
        self.file_folder = file_folder
        self.transform = transform
        self.is_test = is_test
        self.labels = ['BG', 'without_mask', 'with_mask', 'mask_weared_incorrect']

    def __getitem__(self, idx):

        file = self.file_folder[idx]
        img_path = self.img_folder_path + file

        img = Image.open(img_path)
        img = img.convert('RGB')

        if not self.is_test:
            annotation_path = self.annotation_folder_path + file.split('.')[0] + '.xml'
            print(annotation_path)
            with open(annotation_path) as f:
                annotation = f.read()

            n_boxes = len(re.findall('(?<=<xmin>)[0-9]+?(?=</xmin>)', annotation))
            boxes = torch.FloatTensor(self.get_xy(annotation, n_boxes)[0])
            labels = self.get_xy(annotation, n_boxes)[1]
            new_boxes = []

            for k in range(n_boxes):

              new_box = self.box_resize(boxes[k], img)
              new_boxes.append(new_box)

            if self.transform is not None:
                img = self.transform(img)

            return img, new_boxes, labels
        else:
            return img