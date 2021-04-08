'''
BUILDING A CUSTOM DATASET 
FOR TAKING SAMPLES FROM THE DATASET FOR TRAINING
'''

class CustomDataset:
    '''
    THIS CLASS WILL HELP FETCH SAMPLE'S 
    FROM THE DATA WHEN WRAPPED 
    AROUND THE DATALOADER CLASS
    '''
    def __init__(self, data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):
        '''
        NOTE:-
        FOR INDEXING ALWAYS USE ARRAYS 
        THAT WAY IT WILL BE MUCH FASTER 
        '''
        return{
        'sample_features':self.data[idx],
        'sample_targets':self.targets[idx],
        }
