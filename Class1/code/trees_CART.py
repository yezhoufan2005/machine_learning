'''
CART (Classification and Regression Trees) Algorithm
Based on ID3 implementation from Machine Learning in Action
Key features: Binary Tree, Gini Index
'''
import operator

def calcGini(dataSet):
    """计算基尼指数"""
    numEntries = len(dataSet)
    if numEntries == 0: return 0
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini = 1.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        gini -= prob ** 2
    return gini

def splitDataSet(dataSet, axis, value, direction='eq'):
    """划分数据集"""
    retDataSet = []
    for featVec in dataSet:
        if direction == 'eq':
            if featVec[axis] == value:
                retDataSet.append(featVec)
        elif direction == 'neq':
            if featVec[axis] != value:
                retDataSet.append(featVec)
        elif direction == 'lt':
            if featVec[axis] <= value:
                retDataSet.append(featVec)
        elif direction == 'gt':
            if featVec[axis] > value:
                retDataSet.append(featVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """使用基尼指数选择最佳划分特征和划分值"""
    numFeatures = len(dataSet[0]) - 1
    bestGini = 1.0
    bestFeature = -1
    bestSplitValue = None
    isContinuous = False

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        
        # 判断是否为连续值
        current_is_continuous = False
        if isinstance(featList[0], (int, float)) and len(uniqueVals) > 10:
            current_is_continuous = True

        if current_is_continuous:
            # 处理连续值
            sortedUniqueVals = sorted(list(uniqueVals))
            for j in range(len(sortedUniqueVals) - 1):
                splitPoint = (sortedUniqueVals[j] + sortedUniqueVals[j+1]) / 2.0
                sub0 = splitDataSet(dataSet, i, splitPoint, 'lt')
                sub1 = splitDataSet(dataSet, i, splitPoint, 'gt')
                prob0 = len(sub0) / float(len(dataSet))
                prob1 = len(sub1) / float(len(dataSet))
                newGini = prob0 * calcGini(sub0) + prob1 * calcGini(sub1)
                
                if newGini < bestGini:
                    bestGini = newGini
                    bestFeature = i
                    bestSplitValue = splitPoint
                    isContinuous = True
        else:
            # 处理离散值
            for value in uniqueVals:
                sub0 = splitDataSet(dataSet, i, value, 'eq')
                sub1 = splitDataSet(dataSet, i, value, 'neq')
                prob0 = len(sub0) / float(len(dataSet))
                prob1 = len(sub1) / float(len(dataSet))
                newGini = prob0 * calcGini(sub0) + prob1 * calcGini(sub1)
                
                if newGini < bestGini:
                    bestGini = newGini
                    bestFeature = i
                    bestSplitValue = value
                    isContinuous = False
                
    return bestFeature, bestSplitValue, isContinuous

def majorityCnt(classList):
    """多数表决"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """递归构建CART决策树"""
    classList = [example[-1] for example in dataSet]
    # 停止条件
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    
    bestFeat, bestSplitValue, isContinuous = chooseBestFeatureToSplit(dataSet)
    if bestFeat == -1:
        return majorityCnt(classList)
        
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    
    if isContinuous:
        # 连续值
        leftLabel = "<= " + str(bestSplitValue)
        rightLabel = "> " + str(bestSplitValue)
        subLabels = labels[:]
        
        subLeft = splitDataSet(dataSet, bestFeat, bestSplitValue, 'lt')
        subRight = splitDataSet(dataSet, bestFeat, bestSplitValue, 'gt')
    else:
        # 离散值
        leftLabel = "== " + str(bestSplitValue)
        rightLabel = "!= " + str(bestSplitValue)
        subLabels = labels[:]
        
        subLeft = splitDataSet(dataSet, bestFeat, bestSplitValue, 'eq')
        subRight = splitDataSet(dataSet, bestFeat, bestSplitValue, 'neq')

    # 防止无限递归
    if len(subLeft) == 0 or len(subRight) == 0:
        return majorityCnt(classList)

    myTree[bestFeatLabel][leftLabel] = createTree(subLeft, subLabels)
    myTree[bestFeatLabel][rightLabel] = createTree(subRight, subLabels)
            
    return myTree

def classify(inputTree, featLabels, testVec):
    """分类函数"""
    if not isinstance(inputTree, dict):
        return inputTree
        
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    
    # 提取分支条件
    branches = list(secondDict.keys())
    # 找到带有运算符的标签
    for branch in branches:
        if branch.startswith('<= '):
            splitVal = float(branch.split(' ')[1])
            if key <= splitVal:
                return classify(secondDict[branch], featLabels, testVec)
            else:
                for b2 in branches:
                    if b2.startswith('> '):
                        return classify(secondDict[b2], featLabels, testVec)
        elif branch.startswith('== '):
            # 离散值二分
            splitValStr = branch.split('== ')[1]
            # 尝试转换类型
            try:
                if isinstance(key, float): splitVal = float(splitValStr)
                elif isinstance(key, int): splitVal = int(splitValStr)
                else: splitVal = splitValStr
            except:
                splitVal = splitValStr
                
            if key == splitVal:
                return classify(secondDict[branch], featLabels, testVec)
            else:
                for b2 in branches:
                    if b2.startswith('!= '):
                        return classify(secondDict[b2], featLabels, testVec)
                        
    return None
