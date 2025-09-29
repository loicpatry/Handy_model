#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:16:28 2024

@author: patry
"""

import numpy as np

import h5py as h5



def creation_fichier_commoners_rand(EPSILON,nombre_de_tirage,Nombre_annee):
    
   
    f=h5.File('Handy_donnees_full',"w")
    grp1 = f.create_group("Bruit_commoners")
    subgrp11 = grp1.create_group("data variables")
    subgrp12 = grp1.create_group("data de collapse et date")
    
    N=int(Nombre_annee*10)
    
    collapse_fig_1=np.zeros((len(EPSILON),N+1))
    collapse_fig_2=np.zeros((len(EPSILON),N+1))
    collapse_fig_3=np.zeros((len(EPSILON),N+1))
    collapse_fig_4=np.zeros((len(EPSILON),N+1))
    
    r=0
    for epsilon in EPSILON:
        
       
    
  
        
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
                
                
                if x[n]==0 and c==0:
                    collapse_fig_1[r,n]+= 1
                    c=1
        
      
        
        
        commoners_big_fig1=commoners_big
        
        nature_big_fig1=nature_big
        
        ressources_big_fig1=ressources_big
    
    
        
    
        
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
                
                if x[n]==0 and c==0:
                    collapse_fig_2[r,n]+= 1
                    c=1
        
        
    
        commoners_big_fig2=commoners_big
       
        nature_big_fig2=nature_big
       
        ressources_big_fig2=ressources_big
    
    
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
                
                if x[n]==0 and c==0:
                    collapse_fig_3[r,n]+= 1
                    c=1
        
        
        
        commoners_big_fig3=commoners_big
        
        nature_big_fig3=nature_big
        
        ressources_big_fig3=ressources_big
        
    
   
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
    
                if x[n]==0 and c==0:
                    collapse_fig_4[r,n]+= 1
                    c=1
                   
         
    
        commoners_big_fig4=commoners_big
        
        nature_big_fig4=nature_big
        
        ressources_big_fig4=ressources_big
        
        
        
        
        
        eps_grp = subgrp11.create_group(f"epsilon_{r+1}")
                    
        eps_grp.create_dataset("Commoners_big_fig1", data=commoners_big_fig1)
               
        eps_grp.create_dataset("Commoners_big_fig2", data=commoners_big_fig2)
               
        eps_grp.create_dataset("Commoners_big_fig3", data=commoners_big_fig3)
               
        eps_grp.create_dataset("Commoners_big_fig4", data=commoners_big_fig4)
                
        eps_grp.create_dataset("Nature_big_fig1", data=nature_big_fig1)
                
        eps_grp.create_dataset("Nature_big_fig2", data=nature_big_fig2)
        
        eps_grp.create_dataset("Nature_big_fig3", data=nature_big_fig3)
        
        eps_grp.create_dataset("Nature_big_fig4", data=nature_big_fig4)
        
        eps_grp.create_dataset("Ressources_big_fig1", data=ressources_big_fig1)
        
        eps_grp.create_dataset("Ressources_big_fig2", data=ressources_big_fig2)
        
        eps_grp.create_dataset("Ressources_big_fig3", data=ressources_big_fig3)
     
        eps_grp.create_dataset("Ressources_big_fig4", data=ressources_big_fig4)
        
        r=r+1
        
    subgrp12.create_dataset("Collapse_fig_1" , data= collapse_fig_1)
    subgrp12.create_dataset("Collapse_fig_2" , data= collapse_fig_2)
    subgrp12.create_dataset("Collapse_fig_3" , data= collapse_fig_3)
    subgrp12.create_dataset("Collapse_fig_4" , data= collapse_fig_4)
        
   
    return 
nombre_de_tirage=1000
Nombre_annee=1500
EPSILON =np.arange(0,0.4,0.01)

creation_fichier_commoners_rand(EPSILON,nombre_de_tirage,Nombre_annee)
