import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.integrate as sint



#Funções que serão utilizadas:

	#integral indefinida com constante de integração = 0

def integral(f,x,dx=1e-4):
	result = []
	for i in range(len(x)):
		xi=	np.linspace(0,x[i])
		result.append(sint.simps(f(xi),xi,dx=dx))
	return result

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

#importance sample

	#definir quem é g, Ginv:
def g_and_Ginv(x,lambdaa=.05,S=[0,1]):
	#definir constante de normalização, g, e array vaz

	A = 0
	g = np.exp(-lambdaa*x)
	final = []

	#constante de normalização
	A = 1/sint.simps(g,x)

	#multiplicar cada elemento de g por uma constante de normalização
	for i in range(len(g)):
		final.append(g[i]*A)

	#retornar resultado
	return np.array(final)


def MCImpS(f,finv,g,Ginv,N=10,lambdai=.05):

	#Intervalo uniforme [0,1]
	r = np.random.uniform(0,1,N)

	#Calcular inversa de G no intervalo uniforme

	GinI = Ginv(r)

	#calcular f(Ginv)/g(Ginv)

	f_over_g = 0
	for i in range(len(Iu)):
		f_over_g.append(f(GinI)[i]/g(GinI,lambdaa))

	#definir variância
	var = 1/N

	#calcular variância
	var = 1/N*sum(f_over_g)

	return var


















#adicional





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