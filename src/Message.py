from Tag import Tag

class Message:
    def __init__(self, index, tag, subject, body):
        self.index = index
        self.tag = tag
        self.subject = subject
        self.body = body
        self.hamProb = 0
        self.spamProb = 0

    def __str__(self):
        return "%s,%s,%s,%f,%f" % (self.index, self.tag, "spam" if self.isSpam() else "ham", self.hamProb, self.spamProb)

    def __repr__(self):
        return self.__str__()

    def isSpam(self):
        return self.spamProb - self.hamProb > 0

    @staticmethod
    def createFromFile(fileDir, fileName):
        props = fileName.split(".")
        index = props[0]
        tag = props[3]

        file = open(fileDir + fileName)
        subject = file.readline().split(":")[1].strip()
        body = " ".join([line.strip() for line in  file.readlines()])

        msg = Message(index, tag, subject, body)
        return msg

    @staticmethod
    def createBulk(fileDir, fileList):
        msgList = []
        for fileName in fileList:
            msgList.append(Message.createFromFile(fileDir, fileName))
        return msgList
