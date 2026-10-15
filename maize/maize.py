import sys

class node:
    def __init__(self,state,parent,action):
        self.state=state
        self.parent=parent
        self.action=action
class stackfrontier:
    def __init__(self):
        self.stack=[]
    def push(self,node):
        self.stack.append(node)