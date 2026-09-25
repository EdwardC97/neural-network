#this function only tells the nn how far away it was from the correct answer.
#no improvements provided yet

def squared_error(prediction, target):
    difference = prediction - target
    return difference ** 2


