import numpy as np
import matplotlib.pyplot as plt
nombre_de_tirage=1000
Nombre_annee=1500
N=int(Nombre_annee*10)


  
epsilon=0      
       
    
        
fig, ax= plt.subplots(nrows=2,ncols=2,figsize=(12,7))

fig.suptitle('SOCIETE EQUITABLE EULER, dt = 0.1, bruit commoners, epsilon = '+str(epsilon)+', nombre de tirage = '+str(nombre_de_tirage)+'')



commoners_big=np.zeros((N+1,nombre_de_tirage)) 
nature_big=np.zeros((N+1,nombre_de_tirage))
ressources_big=np.zeros((N+1,nombre_de_tirage))


Lambda=np.float32(100) #nature carrying capacity 
beta_c=np.float32(0.03) #birth rate
gamma=np.float32(0.01) #regeneration rate of nature
alpha_M=np.float32(0.07) #famine death rate
s=np.float32(0.0005) #subsistence salary per capita
alpha_m=np.float32(0.01) #normal death rate
rho=np.float32(0.005) #threshold wealth per capita
n=np.float32((alpha_M-beta_c)/(alpha_M-alpha_m)) #some sort of constant
X_m=np.float32((gamma*(Lambda)**2)/(4*n*s)) #Carrying capacity threshold
delta_opt=np.float32(((2*n*s)/Lambda))

                                      ###########DELTA_OPT###############
delta=np.float32(delta_opt)
X=np.float32((gamma/delta)*(Lambda-n*(s/delta))) #Carrying capacity
line_x=[0,Nombre_annee]
line_y=[X/(X_m),X/(X_m)] #Carrying capacity line

x=np.zeros(N+1, dtype='float32')
y=np.zeros(N+1,dtype='float32')
w=np.zeros(N+1,dtype='float32')


x[0]=100
y[0]=100
w[0]=0
t_f=Nombre_annee
t_i=0

#euler integrale 
t=np.linspace(t_i,t_f,N+1,dtype='float32')
dt=(t_f-t_i)/N

for i in range(0,nombre_de_tirage):
    c=0
    commoners_big[0,i]=0
    nature_big[0,i]=1
    ressources_big[0,i]=0

    for n in range(0,N):
	
        if n%10==0:
            R =np.random.normal(0,1)
        w_th=np.single(rho*x[n])
        C=np.single(min(1,w[n]/w_th)*s*x[n])
        alpha=np.single( alpha_m + max(0,1-(C/(s*x[n])))*(alpha_M-alpha_m))
        x[n+1] = x[n] + dt*((beta_c*x[n]-alpha*x[n]) + epsilon*x[n]*R)
        y[n+1] = y[n] + dt*(gamma*y[n]*(Lambda-y[n])-(delta*x[n]*y[n]))  
        w[n+1] = w[n] + dt*((delta*x[n]*y[n])-C) 
    
        if y[n+1]<0: 
            y[n+1]=0
        if x[n+1]<2:
            x[n+1]=0
        if w[n+1]<0:
            w[n+1]=0
        commoners_big[n+1,i]=x[n+1]/(X_m)
        nature_big[n+1,i]=y[n+1]/Lambda
        ressources_big[n+1,i]=w[n+1]/(4*Lambda)
        
   

  


commoners_big_fig1=commoners_big

nature_big_fig1=nature_big

ressources_big_fig1=ressources_big

ressources_variance=np.var(ressources_big,axis=1)
commoners_variance=np.var(commoners_big,axis=1)
nature_variance=np.var(nature_big,axis=1)


  




plt.figure()
ax[0,0].errorbar(t,commoners_big, yerr=commoners_variance, ecolor='lightskyblue',color='blue')
ax[0,0].errorbar(t,nature_big, yerr=nature_variance, ecolor='lawngreen',color='green')
ax[0,0].errorbar(t,ressources_big, yerr=ressources_variance, ecolor='grey',color='black')
ax[0,0].plot(line_x,line_y,color='red',linewidth=0.7)
ax[0,0].set_title(r'$\delta=\delta_{opt*}$')




                                ###########2.5*DELTA_OPT###############


N=int(Nombre_annee*10)

Lambda=np.float32(100) #nature carrying capacity 
beta_c=np.float32(0.03) #birth rate
gamma=np.float32(0.01) #regeneration rate of nature
alpha_M=np.float32(0.07) #famine death rate
s=np.float32(0.0005) #subsistence salary per capita
alpha_m=np.float32(0.01) #normal death rate
rho=np.float32(0.005) #threshold wealth per capita
n=np.float32((alpha_M-beta_c)/(alpha_M-alpha_m)) #some sort of constant
X_m=np.float32((gamma*(Lambda)**2)/(4*n*s)) #Carrying capacity threshold
delta_opt=np.float32(((2*n*s)/Lambda))

commoners_big=np.zeros((N+1,nombre_de_tirage))
nature_big=np.zeros((N+1,nombre_de_tirage))
ressources_big=np.zeros((N+1,nombre_de_tirage))

                                     
delta=np.float32(2.5*delta_opt)
X=np.float32((gamma/delta)*(Lambda-n*(s/delta))) #Carrying capacity
line_x=[0,Nombre_annee]
line_y=[X/(2*X_m),X/(2*X_m)] #Carrying capacity line

x=np.zeros(N+1, dtype='float32')
y=np.zeros(N+1,dtype='float32')
w=np.zeros(N+1,dtype='float32')


x[0]=100
y[0]=100
w[0]=0
t_f=Nombre_annee
t_i=0

    #euler integrale 
t=np.linspace(t_i,t_f,N+1,dtype='float32')
dt=(t_f-t_i)/N    
                                
for i in range(nombre_de_tirage):

    
    c=0
    commoners_big[0,i]=0
    nature_big[0,i]=1
    ressources_big[0,i]=0

    for n in range(0,N):
        if n%10==0:
            R =np.random.normal(0,1)
        w_th=np.single(rho*x[n])
        C=np.single(min(1,w[n]/w_th)*s*x[n])
        alpha=np.single( alpha_m + max(0,1-(C/(s*x[n])))*(alpha_M-alpha_m))
        x[n+1] = x[n] + dt*((beta_c*x[n]-alpha*x[n]) + epsilon*x[n]*R)
        y[n+1] = y[n] + dt*(gamma*y[n]*(Lambda-y[n])-(delta*x[n]*y[n]))  
        w[n+1] = w[n] + dt*((delta*x[n]*y[n])-C) 
    
        if y[n+1]<0: 
            y[n+1]=0
        if x[n+1]<2:
            x[n+1]=0
        if w[n+1]<0:
            w[n+1]=0
        commoners_big[n+1,i]=x[n+1]/(2*X_m)
        nature_big[n+1,i]=y[n+1]/Lambda
        ressources_big[n+1,i]=w[n+1]/(20*Lambda)
        
        



commoners_big_fig2=commoners_big

   
nature_big_fig2=nature_big

ressources_big_fig2=ressources_big
ressources_variance=np.var(ressources_big,axis=1)
commoners_variance=np.var(commoners_big,axis=1)
nature_variance=np.var(nature_big,axis=1)

   
plt.figure()
ax[0,1].errorbar(t,commoners_big, yerr=commoners_variance, ecolor='lightskyblue',color='blue')
ax[0,1].errorbar(t,nature_big, yerr=nature_variance, ecolor='lawngreen',color='green')
ax[0,1].errorbar(t,ressources_big, yerr=ressources_variance, ecolor='grey',color='black')
ax[0,1].plot(line_x,line_y,color='red',linewidth=0.7)

ax[0,1].legend(['carrying capacity','humans', 'nature', 'wealth'], shadow=True)
ax[0,1].set_title(r'$\delta=2.5*\delta_{opt*}$')



                                ###########4*DELTA_OPT###############
   

N=int(Nombre_annee*10)

Lambda=np.float32(100) #nature carrying capacity 
beta_c=np.float32(0.03) #birth rate
gamma=np.float32(0.01) #regeneration rate of nature
alpha_M=np.float32(0.07) #famine death rate
s=np.float32(0.0005) #subsistence salary per capita
alpha_m=np.float32(0.01) #normal death rate
rho=np.float32(0.005) #threshold wealth per capita
n=np.float32((alpha_M-beta_c)/(alpha_M-alpha_m)) #some sort of constant
X_m=np.float32((gamma*(Lambda)**2)/(4*n*s)) #Carrying capacity threshold
delta_opt=np.float32(((2*n*s)/Lambda))

commoners_big=np.zeros((N+1,nombre_de_tirage))
nature_big=np.zeros((N+1,nombre_de_tirage))
ressources_big=np.zeros((N+1,nombre_de_tirage))


delta=np.float32(4*delta_opt)
X=np.float32((gamma/delta)*(Lambda-n*(s/delta))) #Carrying capacity

line_x=[0,Nombre_annee]
line_y=[X/(2*X_m),X/(2*X_m)] #Carrying capacity line

#parameters to choose 
x=np.zeros(N+1, dtype='float32')
y=np.zeros(N+1,dtype='float32')
w=np.zeros(N+1,dtype='float32')


x[0]=100
y[0]=100
w[0]=0
t_f=Nombre_annee
t_i=0

    #euler integrale 
t=np.linspace(t_i,t_f,N+1,dtype='float32')
dt=(t_f-t_i)/N



for i in range(nombre_de_tirage):
    c=0
    commoners_big[0,i]=0
    nature_big[0,i]=1
    ressources_big[0,i]=0

    for n in range(0,N):
        if n%10==0:
            R =np.random.normal(0,1)
        w_th=np.single(rho*x[n])
        C=np.single(min(1,w[n]/w_th)*s*x[n])
        alpha=np.single( alpha_m + max(0,1-(C/(s*x[n])))*(alpha_M-alpha_m))
        x[n+1] = x[n] + dt*((beta_c*x[n]-alpha*x[n]) + epsilon*x[n]*R)
        y[n+1] = y[n] + dt*(gamma*y[n]*(Lambda-y[n])-(delta*x[n]*y[n]))  
        w[n+1] = w[n] + dt*((delta*x[n]*y[n])-C) 
    
        if y[n+1]<0: 
            y[n+1]=0
        if x[n+1]<2:
            x[n+1]=0
        if w[n+1]<0:
            w[n+1]=0
        commoners_big[n+1,i]=x[n+1]/(2*X_m)
        nature_big[n+1,i]=y[n+1]/Lambda
        ressources_big[n+1,i]=w[n+1]/(20*Lambda)
        
  


commoners_big_fig3=commoners_big

nature_big_fig3=nature_big

ressources_big_fig3=ressources_big
ressources_variance=np.var(ressources_big,axis=1)
commoners_variance=np.var(commoners_big,axis=1)
nature_variance=np.var(nature_big,axis=1)

plt.figure()
ax[1,0].errorbar(t,commoners_big, yerr=commoners_variance, ecolor='lightskyblue',color='blue')
ax[1,0].errorbar(t,nature_big, yerr=nature_variance, ecolor='lawngreen',color='green')
ax[1,0].errorbar(t,ressources_big, yerr=ressources_variance, ecolor='grey',color='black')
ax[1,0].plot(line_x,line_y,color='red',linewidth=0.7)
ax[1,0].set_title(r'$\delta=4*\delta_{opt*}$')

  

                                 ###########5.5*DELTA_OPT###############
 

N=int(Nombre_annee*10)

Lambda=np.float32(100) #nature carrying capacity 
beta_c=np.float32(0.03) #birth rate
gamma=np.float32(0.01) #regeneration rate of nature
alpha_M=np.float32(0.07) #famine death rate
s=np.float32(0.0005) #subsistence salary per capita
alpha_m=np.float32(0.01) #normal death rate
rho=np.float32(0.005) #threshold wealth per capita
n=np.float32((alpha_M-beta_c)/(alpha_M-alpha_m)) #some sort of constant
X_m=np.float32((gamma*(Lambda)**2)/(4*n*s)) #Carrying capacity threshold
delta_opt=np.float32(((2*n*s)/Lambda))

commoners_big=np.zeros((N+1,nombre_de_tirage))
nature_big=np.zeros((N+1,nombre_de_tirage))
ressources_big=np.zeros((N+1,nombre_de_tirage))

#Depletion/production factor
delta=np.float32(5.5*delta_opt)
X=np.float32((gamma/delta)*(Lambda-n*(s/delta))) #Carrying capacity

line_x=[0,Nombre_annee]
line_y=[X/(2*X_m),X/(2*X_m)] #Carrying capacity line


#parameters to choose 
x=np.zeros(N+1, dtype='float32')
y=np.zeros(N+1,dtype='float32')
w=np.zeros(N+1,dtype='float32')


x[0]=100
y[0]=100
w[0]=0
t_f=Nombre_annee
t_i=0

#euler integrale 
t=np.linspace(t_i,t_f,N+1,dtype='float32')
dt=(t_f-t_i)/N

for i in range(nombre_de_tirage):

    c=0

    commoners_big[0,i]=0
    nature_big[0,i]=1
    ressources_big[0,i]=0

    for n in range(0,N):
        if n%10==0:
            R =np.random.normal(0,1)

        w_th=np.single(rho*x[n])
        C=np.single(min(1,w[n]/w_th)*s*x[n])
        alpha=np.single( alpha_m + max(0,1-(C/(s*x[n])))*(alpha_M-alpha_m))
        x[n+1] = x[n] + dt*((beta_c*x[n]-alpha*x[n]) + epsilon*x[n]*R)
        y[n+1] = y[n] + dt*(gamma*y[n]*(Lambda-y[n])-(delta*x[n]*y[n]))  
        w[n+1] = w[n] + dt*((delta*x[n]*y[n])-C) 
    
        if y[n+1]<0: 
            y[n+1]=0
        if x[n+1]<2:
            x[n+1]=0
        if w[n+1]<0:
            w[n+1]=0
        commoners_big[n+1,i]=x[n+1]/(2*X_m)
        nature_big[n+1,i]=y[n+1]/Lambda
        ressources_big[n+1,i]=w[n+1]/(20*Lambda)

 
 

commoners_big_fig4=commoners_big

nature_big_fig4=nature_big

ressources_big_fig4=ressources_big

ressources_variance=np.var(ressources_big,axis=1)
commoners_variance=np.var(commoners_big,axis=1)
nature_variance=np.var(nature_big,axis=1)
    



    
plt.figure()
ax[1,1].errorbar(t,commoners_big, yerr=commoners_variance, ecolor='lightskyblue',color='blue')
ax[1,1].errorbar(t,nature_big, yerr=nature_variance, ecolor='lawngreen',color='green')
ax[1,1].errorbar(t,ressources_big, yerr=ressources_variance, ecolor='grey',color='black')
ax[1,1].plot(line_x,line_y,color='red',linewidth=0.7)


ax[1,1].set_title(r'$\delta=5.5*\delta_{opt*}$')

    

