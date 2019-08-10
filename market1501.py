import os

class Market1501():
    def process(self,root,relabel = False):
        files = os.listdir(root)
        filenames = []
        personidset = set()
        for names in files:
            if os.path.splitext(names)[1] != '.jpg':
                continue
            personid = int(os.path.splitext(names)[0].split('_')[0])
            personidset.add(personid)
            if personid == -1:
                continue
            names = root + names
            filenames.append([names,personid])
        id_to_label = {personid:lable for lable,personid in enumerate(personidset)}
        if relabel:
            filenames_ = []
            for filename in filenames:
                filenames_.append([filename[0],id_to_label[filename[1]]])
            return filenames_
        return filenames
    def getinfo(self,imgdex):
        num_personid = []
        num_img = []
        for imgpath in imgdex:
            num_personid.append(imgpath[1])
            num_img.append(imgpath[0])
        num_personid = len(set(num_personid))
        num_img = len(set(num_img))
        return num_personid,num_img
    def __init__(self,root):
        super(Market1501,self).__init__()
        self.root = root
        self.train_dir = self.root + 'bounding_box_train/'
        self.test_dir = self.root + 'bounding_box_test/'
        self.query_dir = self.root + 'query/'
        train = self.process(self.train_dir,True)
        test = self.process(self.test_dir,True)
        query = self.process(self.query_dir,True)
        self.train = train
        self.test = test
        self.query = query
        self.num_train_id,self.num_train_img = self.getinfo(self.train)
        self.num_test_id,self.num_test_img = self.getinfo(self.test)
        self.num_query_id,self.num_query_img = self.getinfo(self.query)