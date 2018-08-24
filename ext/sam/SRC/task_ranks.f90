subroutine task_ranks
	
use grid
implicit none
	
integer i, j, i1, j1
	

!  "Coordinates" of the current subdomain:
	
j = rank/nsubdomains_x + 1
i = rank+1 - (j-1)*nsubdomains_x
	
!  Northen neighbour:
        
i1 = i
j1 = j + 1
if(j1.gt.nsubdomains_y) j1 = 1
ranknn = i1 + (j1-1)*nsubdomains_x - 1

!  North-Eastern neighbour:
        
i1 = i + 1
j1 = j + 1
if(j1.gt.nsubdomains_y) j1 = 1
if(i1.gt.nsubdomains_x) i1 = 1
rankne = i1 + (j1-1)*nsubdomains_x - 1

!  Eastern neighbour:
        
i1 = i + 1
j1 = j 
if(i1.gt.nsubdomains_x) i1 = 1
rankee = i1 + (j1-1)*nsubdomains_x - 1

!  South-Eastern neighbour:
        
i1 = i + 1
j1 = j - 1
if(j1.lt.1) j1 = nsubdomains_y
if(i1.gt.nsubdomains_x) i1 = 1
rankse = i1 + (j1-1)*nsubdomains_x - 1

!  Southern neighbour:
        
i1 = i 
j1 = j - 1
if(j1.lt.1) j1 = nsubdomains_y
rankss = i1 + (j1-1)*nsubdomains_x - 1

!  South-Western neighbour:
        
i1 = i - 1
j1 = j - 1
if(j1.lt.1) j1 = nsubdomains_y
if(i1.lt.1) i1 = nsubdomains_x
ranksw = i1 + (j1-1)*nsubdomains_x - 1

!  Western neighbour:
        
i1 = i - 1
j1 = j 
if(i1.lt.1) i1 = nsubdomains_x
rankww = i1 + (j1-1)*nsubdomains_x - 1

!  North-Western neighbour:
        
i1 = i - 1
j1 = j + 1
if(j1.gt.nsubdomains_y) j1 = 1
if(i1.lt.1) i1 = nsubdomains_x
ranknw = i1 + (j1-1)*nsubdomains_x - 1

end

