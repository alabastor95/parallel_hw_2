#include "boundary.h"
#include <stdio.h>
#include <omp.h>


//grid is parallelised in the x direction

void boundarypsi(double *psi, int m, int n, int b, int h, int w)
{
  //BCs on bottom edge

  for (int i=b+1;i<=b+w-1;i++) {
    psi[i*(n+2)+0] = (double)(i-b);
  }

  for (int i=b+w;i<=m;i++) {
    psi[i*(n+2)+0] = (double)(w);
  }


  //BCS on RHS
  for (int j=1; j <= h; j++) {
    psi[(m+1)*(n+2)+j] = (double) w;
  }


  for (int j=h+1;j<=h+w-1; j++) {
    psi[(m+1)*(n+2)+j]=(double)(w-j+h);
  }
}

void boundaryzet(double *zet, double *psi, int m, int n)
{
  //set top/bottom BCs:
  for (int i=1;i<m+1;i++) {
    zet[i*(n+2)+0]   = 2.0*(psi[i*(n+2)+1]-psi[i*(n+2)+0]);
    zet[i*(n+2)+n+1] = 2.0*(psi[i*(n+2)+n]-psi[i*(n+2)+n+1]);
  }


  //set left BCs:

  for (int j=1;j<n+1;j++) {
    zet[0*(n+2)+j] = 2.0*(psi[1*(n+2)+j]-psi[0*(n+2)+j]);
  }

  //set right BCs

  for (int j=1;j<n+1;j++) {
    zet[(m+1)*(n+2)+j] = 2.0*(psi[m*(n+2)+j]-psi[(m+1)*(n+2)+j]);
  }
}
