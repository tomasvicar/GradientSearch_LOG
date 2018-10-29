clc;clear all;close all;
syms x y t a b c

eqn1 =a ==  cos(t) ^ 2 / (2 * x ^ 2) +  sin(t) ^ 2 / (2 * y ^ 2);
eqn2 =b == - sin(2 * t) / (4 * x ^ 2) + sin(2 * t) / (4 * y ^ 2);
eqn3 =c ==  sin(t) ^ 2 / (2 * x ^ 2) + cos(t) ^ 2 / (2 * y ^2);


sol=solve([eqn1, eqn2, eqn3], [x,y,t]);

xSol = sol.x
ySol = sol.y
tSol = sol.t
