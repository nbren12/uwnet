subroutine task_rank_to_index (rank,i,j)
	
!   returns the pair of  beginning indeces for the subdomain on the  
!   global grid given the subdomain's rank.

use domain

implicit none

integer rank, i, j
		
j = rank/nsubdomains_x 
i = rank - j*nsubdomains_x
	
i = i * (nx_gl/nsubdomains_x) 
j = j * (ny_gl/nsubdomains_y) 
	
end
