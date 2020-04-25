import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.integrate as sint
from time import gmtime

#for slow discrepant distributions
import chaospy as cp 

#date and time for saving files
dttp = gmtime()
forsaving = '{}-{}-{}_{}h{},{}s'.format(dttp[2],dttp[1],dttp[0],dttp[3],dttp[4],dttp[5])
#Funções que serão utilizadas:

	#integral indefinida com constante de integração = 0

def integral(f,x,dx=1e-4,Sx=0):
	result = []
	for i in range(len(x)):
		xi=	np.linspace(Sx,x[i])
		result.append(sint.simps(f(xi),xi,dx=dx))
	return result


#TESTE _+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+

def cos(x):
	return np.cos(x)

def minexp(x):
	return np.exp(-x)

def invfunc(f,x):
	#definir (x,y)
	x1y1 =  np.array([f(x).tolist(),x.tolist()])
	#organizar array usando método .sort():
	
	x1y1_prim = x1y1[x1y1[0,:].sort()][0]
	x1y1_second = x1y1_prim[0][1]
	#return np.array(x1y1[0][0])

def TESTE(f,x):

	x1y1 = np.array([f(x).tolist(),x.tolist()])

	organizador = pd.DataFrame(np.transpose(x1y1),columns='f(x) x'.split())
	Y = organizador.sort_values(by=['f(x)'])['x'].values
	#x1y1 = x1y1[x1y1[:,0].sort()][0]

	return [x1y1,Y]





#TESTE _+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+

#Implementação do método


	#definição das constantes
a, b = .3039110,.11703832


	#definição da função a ser integrada
def f(x,a=.11703832,b=.3039110,n=1):
	F=0
	if n ==1:
		F=np.exp(-a*x)*np.cos(b*x)
	else:
		F=(np.exp(-a*x)*np.cos(b*x))**n
	return F


	#Método cru

def MCint(f,a=0,b=1,N=10,seed=False):
	"""
		f = função
		a,b = float, float - numeros da atividade
		N = tamanho da amostra
		seed = False ou numero inteiro: gerador de numero aleatório (PRNG).

		Devolve DataFrame com colunas {'Resultado','Média','Média Quadrática','Variância','Erro','Steps'}
	"""
	I, U, mean, meanquad, var, err = 0, 0, 0, 0, 0, 0

	if seed == False:
		U = np.random.uniform(a,b,N)
	elif seed == False:
		print('Definir seed com um int ou False para experimentos aleatórios.')
	else:
			#experimento com semente aleatória escolhida.
		np.random.seed(seed)
		U =np.random.uniform(a,b,N)

		#calcular média, média quadrática, variância e erro
	mean, meanquad = sum(f(U))/N, sum(f(U,n=2)/N)
	var = meanquad-mean**2
	err = (var/N)**.5


		#Aplicar método	
	I=((b-a)/N)*sum(f(U))

		#salvar resultados em um DataFrame
	return pd.DataFrame(np.array([['{} +/- {}'.format(I,err), mean, meanquad, var, err, int(N)]]),index=['Valores'],columns=['Resultado','Média','Média Quadrática','Variância','Erro','Passos'])

	#Calcular com precisão

def MCcrude(f,a=0,b=1,Ni=10,precisao=.005,seed=False,log=True):
	"""
		f = função def f(x);
		a,b = float, float - numeros da atividade;
		Ni = Tamanho inicial da amostra (int);
		seed = False ou numero inteiro: semente aleatória;
		log = True or False: printa a relação entre passos x erro;

		Devolve DataFrame com colunas {'Resultado','Média','Média Quadrática','Variância','Erro','Steps'}.
	"""

#(f,a=0,b=1,Ni=10,precisao=.005,seed=False):
#variáveis iniciais

	N, U = Ni, MCint(f,N=Ni,seed=seed)
	erro = float(U['Erro'].values[0])

	#definir laço: enquanto erro > precisao somar +1 no N e imprimir os passos e o erro associado
	if log==True:
		while erro > precisao:
			erro=float(MCint(f,a,b,N,seed)['Erro'].values[0])
			N=N+1
			print('passos: {},\n erro: {}'.format(N,erro))
	elif log==False:
		while erro > precisao:
			erro=float(MCint(f,a,b,N,seed)['Erro'].values[0])
			N=N+1
	else:
		print('log=True or False')

	#Usar função MCint() para criar um dataframe que será o output
	F = MCint(f,a,b,N,seed)
	return pd.concat([F,pd.DataFrame(np.array([precisao]),columns=['Acurácia'],index=['Valores'])],axis=1)


#Amostragem de importancia

	#definir quem é g, Ginv:
def g_and_Ginv(x,lambdaa=.05):
	#definir constante de normalização, g, e array vazio

	A = 0
	g = np.exp(-lambdaa*x)
	final = []
	G = []
	integral_of_g, Ginv = [], []

	#constante de normalização
	A = 1/sint.simps(g,x)

	#multiplicar cada elemento de g por uma constante de normalização
	for i in range(len(g)):
		final.append(g[i]*A)

	#final = final.append(final[49])
	#G = integral(np.array(final),x)


	#definir a inversa de G que estará definida em [0,1]

	Ginv = -(1/lambdaa)*np.log((np.exp(-lambdaa*b)-np.exp(-lambdaa*a))*x+np.exp(-lambdaa*a))
	#z =  -(1/lambdaa)*np.log(-lambdaa*x/A+np.exp(-lambdaa*a))
	Ginverso = []

	#for i in range(len(x)):
	#	Ginverso.append(Ginv(x[i]))
	#y = np.array(Ginverso)

	return [final,Ginv]



def MCImpS(f,Ginv,N=10,lambdai=.05,tests=100,seed=False):

	#Intervalo uniforme [0,1]
	if seed == False:
		r = np.random.uniform(0,1,N)
	elif seed==True:
		r = np.random.uniform(0,1,N)
	else:
		np.seed(seed)
		r = np.random.uniform(0,1,N)

	variancia = []
	lambdas = [i*.05 for i in range(1,tests+1)]
	

	for lamb in lambdas:

		#calcular f(Ginv)/g(Ginv)
		
		GinI = Ginv(r,lamb)[1]
		F = f(GinI)
		g = Ginv(GinI,lamb)[0]

		f_over_g = []
		f_over_g2 = []


		for i in range(len(r)):
			f_over_g.append(F[i]/g[i])
			f_over_g2.append((F[i]/g[i])**2)

		#definir variância, média e média quadrática
		var = 0
		mean = 0
		meanquad = 0

		#calcular média
		mean = 1/N*sum(f_over_g)
		
		#calcular média quadrática

		meanquad = 1/N*sum(f_over_g2)

		#calcular variância
		var = meanquad-mean**2
		variancia.append(var)


	U = [variancia, lambdas]

	#Escolher menor lambda
	df=pd.DataFrame(np.transpose(U),columns='variância lambdas'.split())
	df2 = df
	df = df[df['variância']==df['variância'].min()]

	minlambda = df['lambdas'].values[0]

	#Aplicar o método
	Gg = Ginv(r,minlambda)[1]
	F = f(Gg)
	G = Ginv(Gg,minlambda)[0]

	f_over_gfinal = [F[i]/G[i] for i in range(len(Gg))]

	I=1/N*sum(f_over_gfinal)
	err = 1/N*(sum([(F[i]/G[i])**2 for i in range(len(Gg))])-I**2)**.5

	return I, err

	#return I,r,minlambda, Gg, F, G, df2, df 



#hit and miss


def hit_and_miss(f,a=0,b=1,N=1000,graph=False,log=False,seed=False):
	"""
		f = função def f(x);
		a,b = float, float - intervalo;
		Ni = Numero de pontos
		graph = Booleano, gera o gráfico do experimento
		log = True or False: printa a relação entre passos x erro;
		seed = False ou numero inteiro: semente aleatória;

		Devolve DataFrame com colunas {'Resultado','Média','Média Quadrática','Variância','Erro','Steps'}.
	"""

	Nstar = 0
	xy = [[],[]]
	
	#definir semente aleatória ou não
	if seed!=False:
		print('Semente aleatória indefinida')
	else:
		np.random.seed(seed)

	X =np.random.uniform(a,b,N)
	H = max(f(X))
	Y = np.random.uniform(0,H,N)
	

	#gerar pontos aleatórios e verificar se pertencem ao (x,y)<=(x,f(x)) (função h(x,y) sendo integrada em D**2)
	for i in range(len(X)):
		if ([X[i],Y[i]]<=[X[i],f(X)[i]] and Y[i]>0):
			Nstar+=1
			xy[0].append(X[i])
			xy[1].append(Y[i])
			if log:
				logs=N-Nstar
				print('{} de {} tentativas.'.format(logs,N))
				print('{} acertos'.format(Nstar))
	
	#Cálculos necessários

	#Média = 1/N*Σxn == Nstar/N = P
	mean = Nstar/N

	#Integral approx (b-a)*H*Nstar/N == (b-a)*H*mean
	I = (b-a)*H*mean
	#variância
	var = mean - mean**2
	err = (var/N)**.5
	Ie = I*err
	resultado = pd.DataFrame(np.array([['{} +/- {}'.format(I,Ie),var,err,H,N,Nstar]]),columns='Resultado Variância Erro H N N*'.split(),index='Valores'.split())

	#criar gráfico: só será criado caso graph == True
	if graph:
		fig, ax = plt.subplots(figsize=(10,8))
		ax.plot(X,Y,marker='o',markersize=8,lw=0,color='#008080',alpha=.5,label='Pontos')
		ax.plot(xy[0],xy[1],marker='x',markersize=3,lw=0,color='#fca6ea',label='Erro')
		x=np.arange(a,b,1e-6)
		ax.plot(x,f(x),color='#f7347a',lw=3)
		ax.set_xlabel('x')
		ax.set_ylabel('f(x)')
		ax.set_title('Acertos e erros.')
		ax.legend(shadow=True,loc=2)
		ax.spines['left'].set_position('zero')
		ax.spines['right'].set_color('none')
		ax.spines['bottom'].set_position('zero')
		ax.spines['top'].set_color('none')
		fig.show()
		fig.savefig('fig'+forsaving,format='png',dpi=330,quality=95)
		return resultado, fig
	else:
		return resultado




def ajustarerro(f,a=0,b=1,Ni=100,precisao=.01,graph=False,log=False,seed=False):
	N, U = Ni, hit_and_miss(f,a,b,Ni,graph,log,seed)
	erro = float(U['Erro'].values[0])
	F = U
	ti = [dttp[3],dttp[4],dttp[5]]
	#definir laço: enquanto erro > precisao somar +1 no N e imprimir os passos e o erro associado
	print('Processo iniciado em {}h{}min{}s'.format(ti[0],ti[1],ti[2]))
	while erro > precisao:
		F = hit_and_miss(f,a,b,N,graph,seed)
		erro=float(F['Erro'].values[0])
		N=N+1
		if log==True:
			print('passos: {},\n erro: {}'.format(N,erro))


	tf = [dttp[3],dttp[4],dttp[5]]
	Dt = list(np.array(tf)-np.array(ti))
	if Dt[0]==0:
		print('{}min{}s para calcular'.format(Dt[1],Dt[2]))
	if [Dt[0],Dt[1]]==[0,0]:
		print('{}s para calcular'.format(Dt[2]))
	
	print('tf = {}h{}min{}s'.format(Dt[0],Dt[1],Dt[2]))

	return pd.concat([F,pd.DataFrame(np.array([precisao]),columns=['Acurácia'],index=['Valores'])],axis=1)






#adicional
#importance sample





def varc(f,x,n):

	return 1/n*sint.simps()

g = np.random.random_sample(100)

def aproxxint(interval):
	summ=sum(f(interval))
	return summ*((max(interval)-min(interval))/len(interval))


def meanint(x,I):
	return 1/(I[len(I)-1]-min(I[0]))


def randomplot(N,a=1):
	for i in range(N):
		x=np.random.uniform(-a,a,i)
		plt.scatter(x,np.cos(x),label='{}'.format(i))
		plt.legend()