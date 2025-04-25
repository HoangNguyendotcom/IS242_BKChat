import sys
sys.path.append("/Users/mac/Documents/Master_k24/242_Intelligent_Systems/IS242_BKChat/backend")
from app import mongo
from run import app

def init_mislabeled_collections():
    with app.app_context():
        # Create toxic_messages collection if it doesn't exist
        if 'toxic_messages' not in mongo.db.list_collection_names():
            mongo.db.create_collection('toxic_messages')
            print("Created toxic_messages collection")
        
        # Create not_toxic_messages collection if it doesn't exist
        if 'not_toxic_messages' not in mongo.db.list_collection_names():
            mongo.db.create_collection('not_toxic_messages')
            print("Created not_toxic_messages collection")
        
        # Create indexes for better query performance
        mongo.db.toxic_messages.create_index([("messageId", 1)])
        mongo.db.toxic_messages.create_index([("userId", 1)])
        mongo.db.not_toxic_messages.create_index([("messageId", 1)])
        mongo.db.not_toxic_messages.create_index([("userId", 1)])
        
        print("Created indexes for mislabeled messages collections")

if __name__ == "__main__":
    init_mislabeled_collections() 