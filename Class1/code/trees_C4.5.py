'''
C4.5 Decision Tree Algorithm
Based on ID3 implementation from Machine Learning in Action
'''
from math import log
import operator

def calcShannonEnt(dataSet):
    """计算熵"""
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value, direction='eq'):
    """划分数据集"""
    retDataSet = []
    for featVec in dataSet:
        if direction == 'eq':
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        elif direction == 'lt':
            if featVec[axis] <= value:
                retDataSet.append(featVec) # 连续值不删除特征
        elif direction == 'gt':
            if featVec[axis] > value:
                retDataSet.append(featVec) # 连续值不删除特征
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """使用信息增益率选择最佳划分特征"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestGainRatio = 0.0
    bestFeature = -1
    bestSplitPoint = None

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        
        # 判断是否为连续值
        isContinuous = False
        if isinstance(featList[0], (int, float)) and len(uniqueVals) > 10:
            isContinuous = True

        if isContinuous:
            # 处理连续值
            sortedUniqueVals = sorted(list(uniqueVals))
            for j in range(len(sortedUniqueVals) - 1):
                splitPoint = (sortedUniqueVals[j] + sortedUniqueVals[j+1]) / 2.0
                
                # 计算在该分裂点下的信息增益
                subDataSet0 = splitDataSet(dataSet, i, splitPoint, 'lt')
                subDataSet1 = splitDataSet(dataSet, i, splitPoint, 'gt')
                prob0 = len(subDataSet0) / float(len(dataSet))
                prob1 = len(subDataSet1) / float(len(dataSet))
                
                newEntropy = prob0 * calcShannonEnt(subDataSet0) + prob1 * calcShannonEnt(subDataSet1)
                infoGain = baseEntropy - newEntropy
                
                # 计算分裂信息
                splitInfo = 0.0
                if prob0 > 0: splitInfo -= prob0 * log(prob0, 2)
                if prob1 > 0: splitInfo -= prob1 * log(prob1, 2)
                
                gainRatio = infoGain / splitInfo if splitInfo != 0 else 0
                
                if gainRatio > bestGainRatio:
                    bestGainRatio = gainRatio
                    bestFeature = i
                    bestSplitPoint = splitPoint
        else:
            # 处理离散值
            newEntropy = 0.0
            splitInfo = 0.0
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value, 'eq')
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
                if prob > 0: splitInfo -= prob * log(prob, 2)
            
            infoGain = baseEntropy - newEntropy
            gainRatio = infoGain / splitInfo if splitInfo != 0 else 0
            
            if gainRatio > bestGainRatio:
                bestGainRatio = gainRatio
                bestFeature = i
                bestSplitPoint = None
                
    return bestFeature, bestSplitPoint

def majorityCnt(classList):
    """多数表决"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """递归构建C4.5决策树"""
    classList = [example[-1] for example in dataSet]
    # 停止条件
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    
    bestFeat, bestSplitPoint = chooseBestFeatureToSplit(dataSet)
    if bestFeat == -1:
        return majorityCnt(classList)
        
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    
    if bestSplitPoint is not None:
        # 连续值分支
        leftLabel = "<= " + str(bestSplitPoint)
        rightLabel = "> " + str(bestSplitPoint)
        
        # 连续值不删除特征
        myTree[bestFeatLabel][leftLabel] = createTree(splitDataSet(dataSet, bestFeat, bestSplitPoint, 'lt'), labels[:])
        myTree[bestFeatLabel][rightLabel] = createTree(splitDataSet(dataSet, bestFeat, bestSplitPoint, 'gt'), labels[:])
    else:
        # 离散值分支
        newLabels = labels[:bestFeat] + labels[bestFeat+1:]
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value, 'eq'), newLabels[:])
            
    return myTree

def classify(inputTree, featLabels, testVec):
    """分类函数"""
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    
    # 检查是否为连续值分支
    isContinuous = False
    for k in secondDict.keys():
        if isinstance(k, str) and (k.startswith('<=') or k.startswith('>')):
            isContinuous = True
            break
            
    if isContinuous:
        # 提取分裂点
        for k in secondDict.keys():
            if k.startswith('<='):
                splitPoint = float(k.split(' ')[1])
                if key <= splitPoint:
                    valueOfFeat = secondDict[k]
                else:
                    for k2 in secondDict.keys():
                        if k2.startswith('>'):
                            valueOfFeat = secondDict[k2]
                            break
                break
    else:
        if key in secondDict:
            valueOfFeat = secondDict[key]
        else:
            return None

    if isinstance(valueOfFeat, dict):
        return classify(valueOfFeat, featLabels, testVec)
    else:
        return valueOfFeat
