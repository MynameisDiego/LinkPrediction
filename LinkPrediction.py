import pandas as pd
from sklearn.model_selection import train_test_split

import networkx as nx
import numpy as np
import time
import copy
import os

from collections import defaultdict

def minimizedKatz(H,nodeH,c):
    Adj = nx.adjacency_matrix(H, weight=None)
    w, v = np.linalg.eigh(Adj.toarray())
    lambdaAdj = max([abs(x) for x in w])
    I = np.eye(Adj.shape[0])
    S = None
    S = np.linalg.inv(I - c/lambdaAdj * Adj)
    nodosTotal,dfk,indexTotal = [],[],[]
    
    for n in nodeH:
        for j in nodeH:
           nodosTotal.append([n,j])     
    cuenta = 0
    
    for i in range(len(S)):
        for j in range(len(S)):
            df5 = [nodosTotal[cuenta][0],nodosTotal[j][1],S[i,j]]
            dfk.append(df5)
            cuenta += 1    
            
    for i in range(len(dfk)):
        if(dfk[i][0] == dfk[i][1] or dfk[i][2] == 0 or H.has_edge(dfk[i][0],dfk[i][1])):
            indexTotal.append(i)
    
    dfK = pd.DataFrame(data = dfk, columns = ['N1','N2','Score'])
    dfK = dfK.drop(indexTotal)
    dfK = dfK.reset_index(drop = True)

    return dfK


def _is_converge(s1, s2, eps=1e-4):
  for i in s1.keys():
    for j in s1[i].keys():
      if abs(s1[i][j] - s2[i][j]) >= eps:
        return False
    return True

def SimRank(H,nodeH,c):
    sim_old = defaultdict(list)
    sim = defaultdict(list)
    for n in nodeH:
        sim[n] = defaultdict(int)
        sim[n][n] = 1
        sim_old[n] = defaultdict(int)
        sim_old[n][n] = 0
    
    for iter_ctr in range(1):
        if _is_converge(sim, sim_old):
          break
        sim_old = copy.deepcopy(sim)
        for i, u in enumerate(nodeH):
          for v in nodeH:
            if u == v:
              continue
            s_uv = 0.0
            for n_u in H.neighbors(u):
              for n_v in H.neighbors(v):
                s_uv += sim_old[n_u][n_v]
            sim[u][v] = (c * s_uv / (len(H.neighbors(u)) * len(H.neighbors(v))))\
            if len(H.neighbors(u)) * len(H.neighbors(v)) > 0 else 0
    
    dfS = []
    for n in nodeH:
        for n2 in nodeH:
            if H.has_edge(n,n2) == True or n == n2 or sim[n][n2] is 0 :
                continue
            else:
                df4 = df4 =[n,n2,sim[n][n2]]
                dfS.append(df4)
    return dfS


def minimizedCN(H,nodeH,dataTestEntrenamiento):
    df = []
    for n in nodeH:
        for n2 in nodeH:
            if H.has_edge(n,n2) == True or n == n2 or len(sorted(nx.common_neighbors(H, n, n2))) == 0 :
                continue
            else:
                df.append([n,n2,len(list(nx.common_neighbors(H, n, n2)))])  
    return df

def minimizedAdamic(nodeH,dataTestEntrenamiento,H,dfAdamic):
    dfA = []
    for n in nodeH:
        for n2 in nodeH:
            if H.has_edge(n,n2) == True or n == n2 or sorted(nx.adamic_adar_index(H, [(n, n2)]))[0][2] is 0 :
                continue
            else:
                dfA.append([n,n2,sorted(nx.adamic_adar_index(H, [(n, n2)]))[0][2]])
    return dfA

def tester(dfT,dataTestEntrenamiento):
    
    dfT = pd.DataFrame(data = dfT, columns = ['N1','N2','Score'])
    aciertos = len(dfT.merge(dataTestEntrenamiento, how='inner', on = ['N1','N2']))
    precision = aciertos/len(dataTestEntrenamiento) 
    print(precision)
    return precision


def AUC(datav,dataTestEntrenamiento):
    DataVector = pd.DataFrame(data = datav, columns = ['N1','N2','Score'])

    acertados = (DataVector.merge(dataTestEntrenamiento, how='inner', on = ['N1','N2']))


    general = pd.concat([DataVector,acertados])

    general = general.drop_duplicates(keep = False)


    n = int((acertados.size) * 0.1)+ 1
    
    gEscogidos = general.sample(n = n)
    aEscogidos = acertados.sample(n = n)
    
    generalContador = 0
    acertadoContador = 0
    
    for i in range(n):
        if (aEscogidos.iat[i,2] > gEscogidos.iat[i,2]):
            acertadoContador = acertadoContador + 1
        else:
            generalContador = generalContador + 1
    
    auc = ((acertadoContador + (generalContador * 0.5))/n)
    return auc

def Testing(loops,address,debugger,text): 
    text.write("DataTest: " + address + "\n") 
    text.write("================================================" + "\n") 
    if(debugger):
        print("Leyendo data de ", address)
    dataTest = pd.read_csv(address, delim_whitespace=True, header=None, names=['N1','N2'])
    dataTest = pd.DataFrame(data = dataTest)
    DataE, DataT = train_test_split(dataTest,random_state = 22, test_size = 0.2)
    #DataE.to_csv('d:/Users/Diego/Downloads/Desktop/DataE.txt', index=False, sep=' ', header=None)
    #DataT.to_csv('d:/Users/Diego/Downloads/Desktop/DataT.txt', index=False, sep=' ', header=None)
    dataTestEntrenamiento = pd.DataFrame(data = DataT)
    H = nx.from_pandas_dataframe(DataE,'N1','N2')
    nodeH = H.nodes(data=False)    
    
    
    #==========================Common N======================================
    
    
    cn = [] #Arreglo de los intentos de CN
    cnResults = [] #Arreglo de los intentos de CN(RESULTADOS)
    if(debugger):
        print("Corriendo CD")
    for n in range(loops):
        if(debugger):
            print("Loop de CN ", n + 1)
        stime = time.time()
        df = minimizedCN(H,nodeH,dataTestEntrenamiento)
        ftime = time.time() - stime #Obtengo el tiempo demorado
        print("El tiempo de ejecucion de entrenamiento es: ",round(ftime,3),"segundos")
        text.write("Tiempo Entrenamiento Common Neighbors: " + str(round(ftime,3)) + " segundos" + "\n") 
        stime = time.time()
        result = tester(df,dataTestEntrenamiento) 
        ftime = time.time() - stime #Obtengo el tiempo demorado
        print("El tiempo de ejecucion de Testeo es: ",round(ftime,3),"segundos")
        text.write("Tiempo Testeo Common Neighbors: " + str(round(ftime,3)) + " segundos" + "\n") 
        cn.append(df)
        cnResults.append(result)
        print("Calculando AUC")
        AUCc = AUC(df,dataTestEntrenamiento)
        text.write("AUC Common Neighbors: " + str(AUCc) + "\n") 
      
    #==========================Adamic D======================================
          
    ad = [] #Arreglo de los intentos de AK
    adResult = []
    for n in range(loops):
        if(debugger):
            print("Loop de AD ", n + 1)
        stime = time.time()
        dfAdamic = []
        dfA = minimizedAdamic(nodeH,dataTestEntrenamiento,H,dfAdamic)
        ftime = time.time() - stime #Obtengo el tiempo demorado
        print("El tiempo de ejecucion de entrenamiento es: ",round(ftime,3),"segundos")
        text.write("Tiempo Entrenamiento Adamic-Adar: " + str(round(ftime,3)) + " segundos" + "\n") 
        stime = time.time()
        result1 = tester(dfA,dataTestEntrenamiento) 
        ftime = time.time() - stime #Obtengo el tiempo demorado
        print("El tiempo de ejecucion de Testeo es: ",round(ftime,3),"segundos")
        text.write("Tiempo Testeo Adamic-Adar: " + str(round(ftime,3)) + " segundos" + "\n") 
        adResult.append(result1)
        
        
        ad.append(dfA)
        print("Calculando AUC")
        AUCa = AUC(dfA,dataTestEntrenamiento)
        text.write("AUC Adamic Adar: " + str(AUCa) + "\n") 
    #==========================KaTz======================================
    
    
    katz = [] #Arreglo de los intentos de CN
    katzResults = [] #Arreglo de los intentos de CN(RESULTADOS)
    if(debugger):
        print("Corriendo Katz")
    for n in range(loops):
        if(debugger):
            print("Loop de Katz ", n + 1)
        stime = time.time()
        dfK = minimizedKatz(H,nodeH,0.9)
        print("El tiempo de ejecucion de entrenamiento es: ",round(ftime,3),"segundos")
        text.write("Tiempo Entrenamiento Katz: " + str(round(ftime,3)) + " segundos" + "\n") 
        stime = time.time()
        result2 = tester(dfK,dataTestEntrenamiento) 
        ftime = time.time() - stime #Obtengo el tiempo demorado
        print("El tiempo de ejecucion de Testeo es: ",round(ftime,3),"segundos")
        text.write("Tiempo Entrenamiento Katz: " + str(round(ftime,3)) + " segundos" + "\n") 
        katz.append(dfK)
        katzResults.append(result2)
        print("Calculando AUC")
        AUCk = AUC(dfK,dataTestEntrenamiento)
        text.write("AUC Katz: " + str(AUCk) + "\n") 


def main():
    fileDirs = [line.rstrip() for line in open('Files.txt')]
    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    i = 1
    for dir in fileDirs:
        print(dir)
        file = open("./Results/Data" + str(i) + ".txt","w") 
        Testing(1,dir,True,file)
        i += 1
        file.close() 
    pass

if __name__ == "__main__":
    main()
    