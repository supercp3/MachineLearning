import matplotlib.pyplot as plt 
'''
绘制树形图
'''
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
	createPlot.axl.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords="axes fraction",va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

'''
def createPlot():
	fig=plt.figure(1,facecolor="white")
	fig.clf()
	createPlot.axl=plt.subplot(111,frameon=False)
	plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)
	plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode)
	plt.show()
'''
#获得叶子节点的个数：递归思想
def getNumLeafs(myTree):
	numLeafs=0 
	firstStr=list(myTree.keys())[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs+=getNumLeafs(secondDict[key])
		else:
			numLeafs+=1 
	return numLeafs 
#获得树的深度
def getTreeDepth(myTree):
	maxDepth=0 
	firstStr=list(myTree.keys())[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth=1+getTreeDepth(secondDict[key])
		else:
			thisDepth=1 
		if thisDepth>maxDepth:
			maxDepth=thisDepth
	return maxDepth 

def retrieveTree(i):
	listOfTrees=\
	[{'no surfacing':{0:'no',1:{'flippers':
		{0:'no',1:'yes'}}}},
	{'no surfacing':{0:'no',1:{'flippers':
	{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}
	]
	return listOfTrees[i]
#填充子节点和父节点之间的内容：0/1
def plotMidText(cntrPt,parentPt,txtString):
	xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
	yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
	createPlot.axl.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
	numLeafs=getNumLeafs(myTree)
	depth=getTreeDepth(myTree)
	firstStr=list(myTree.keys())[0]
	cntrPt=(plotTree.xoff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yoff)
	plotMidText(cntrPt,parentPt,nodeTxt)
	plotNode(firstStr,cntrPt,parentPt,decisionNode)
	secondDict=myTree[firstStr]
	plotTree.yoff=plotTree.yoff-1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.xoff=plotTree.xoff+1.0/plotTree.totalW 
			plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),cntrPt,leafNode)
			plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
	plotTree.yoff=plotTree.yoff+1.0/plotTree.totalD

def createPlot(inTree):
	fig=plt.figure(1,facecolor='white')
	fig.clf()
	axprops=dict(xticks=[],yticks=[])
	createPlot.axl=plt.subplot(111,frameon=False,**axprops)
	plotTree.totalW=float(getNumLeafs(inTree))
	plotTree.totalD=float(getTreeDepth(inTree))
	plotTree.xoff=-0.5/plotTree.totalW
	plotTree.yoff=1.0 
	plotTree(inTree,(0.5,1.0),'')
	plt.show()


'''
myTree=retrieveTree(1)
print(myTree)
createPlot(myTree)
'''