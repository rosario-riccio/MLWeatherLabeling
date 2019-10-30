"""This file contains the class ManageDB to manage every db operations"""
import sys
from pymongo import MongoClient

class ManageDB(object):
    def __init__(self):
        """This is constructor"""
        try:
            self.client = MongoClient("mongodb://localhost:27017/")
            db = self.client.MediStormSeekerDB
        except Exception as e:
            print("DB not ready " + str(e))
        self.db = db
        print("DB started, there are n. polygon: ",self.db.PolygonCollection.count())


    #---------------------------LABELING----------------------------------

    def countLabelDB(self):
        """This method gets the number of label"""
        count = self.db.LabelCollection.find().count()
        print("number Label:",count)
        return count

    def listLabelDB(self):
        """This method asks the DB the labels' list"""
        count = self.db.LabelCollection.find().count()
        print("Number Label: ",count)
        if(count > 0):
            result = True
            cursorListLabel = self.db.LabelCollection.find()
            return result,cursorListLabel
        else:
            result = False
            cursorListLabel = None
            return result,cursorListLabel


try:
    managedb = ManageDB()
except Exception as e:
    print("errore: DB not Ready")
    sys.exit(0)
