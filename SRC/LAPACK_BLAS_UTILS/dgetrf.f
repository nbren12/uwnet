      SUBROUTINE DGETF2( M, N, A, LDA, IPIV, INFO )
*
*  -- LAPACK routine (version 3.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LDA, M, N
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      DOUBLE PRECISION   A( LDA, * )
*     ..
*
*  Purpose
*  =======
*
*  DGETF2 computes an LU factorization of a general m-by-n matrix A
*  using partial pivoting with row interchanges.
*
*  The factorization has the form
*     A = P * L * U
*  where P is a permutation matrix, L is lower triangular with unit
*  diagonal elements (lower trapezoidal if m > n), and U is upper
*  triangular (upper trapezoidal if m < n).
*
*  This is the right-looking Level 2 BLAS version of the algorithm.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix A.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the m by n matrix to be factored.
*          On exit, the factors L and U from the factorization
*          A = P*L*U; the unit diagonal elements of L are not stored.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,M).
*
*  IPIV    (output) INTEGER array, dimension (min(M,N))
*          The pivot indices; for 1 <= i <= min(M,N), row i of the
*          matrix was interchanged with row IPIV(i).
*
*  INFO    (output) INTEGER
*          = 0: successful exit
*          < 0: if INFO = -k, the k-th argument had an illegal value
*          > 0: if INFO = k, U(k,k) is exactly zero. The factorization
*               has been completed, but the factor U is exactly
*               singular, and division by zero will occur if it is used
*               to solve a system of equations.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   SFMIN 
      INTEGER            I, J, JP
*     ..
*     .. External Functions ..
      DOUBLE PRECISION   DLAMCH      
      INTEGER            IDAMAX
      EXTERNAL           DLAMCH, IDAMAX
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGER, DSCAL, DSWAP, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      IF( M.LT.0 ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -4
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DGETF2', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( M.EQ.0 .OR. N.EQ.0 )
     $   RETURN
*
*     Compute machine safe minimum 
* 
      SFMIN = DLAMCH('S')  
*
      DO 10 J = 1, MIN( M, N )
*
*        Find pivot and test for singularity.
*
         JP = J - 1 + IDAMAX( M-J+1, A( J, J ), 1 )
         IPIV( J ) = JP
         IF( A( JP, J ).NE.ZERO ) THEN
*
*           Apply the interchange to columns 1:N.
*
            IF( JP.NE.J )
     $         CALL DSWAP( N, A( J, 1 ), LDA, A( JP, 1 ), LDA )
*
*           Compute elements J+1:M of J-th column.
*
            IF( J.LT.M ) THEN 
               IF( ABS(A( J, J )) .GE. SFMIN ) THEN 
                  CALL DSCAL( M-J, ONE / A( J, J ), A( J+1, J ), 1 ) 
               ELSE 
                 DO 20 I = 1, M-J 
                    A( J+I, J ) = A( J+I, J ) / A( J, J ) 
   20            CONTINUE 
               END IF 
            END IF 
*
         ELSE IF( INFO.EQ.0 ) THEN
*
            INFO = J
         END IF
*
         IF( J.LT.MIN( M, N ) ) THEN
*
*           Update trailing submatrix.
*
            CALL DGER( M-J, N-J, -ONE, A( J+1, J ), 1, A( J, J+1 ), LDA,
     $                 A( J+1, J+1 ), LDA )
         END IF
   10 CONTINUE
      RETURN
*
*     End of DGETF2
*
      END
      SUBROUTINE DGETRF( M, N, A, LDA, IPIV, INFO )
*
*  -- LAPACK routine (version 3.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LDA, M, N
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      DOUBLE PRECISION   A( LDA, * )
*     ..
*
*  Purpose
*  =======
*
*  DGETRF computes an LU factorization of a general M-by-N matrix A
*  using partial pivoting with row interchanges.
*
*  The factorization has the form
*     A = P * L * U
*  where P is a permutation matrix, L is lower triangular with unit
*  diagonal elements (lower trapezoidal if m > n), and U is upper
*  triangular (upper trapezoidal if m < n).
*
*  This is the right-looking Level 3 BLAS version of the algorithm.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix A.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the M-by-N matrix to be factored.
*          On exit, the factors L and U from the factorization
*          A = P*L*U; the unit diagonal elements of L are not stored.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,M).
*
*  IPIV    (output) INTEGER array, dimension (min(M,N))
*          The pivot indices; for 1 <= i <= min(M,N), row i of the
*          matrix was interchanged with row IPIV(i).
*
*  INFO    (output) INTEGER
*          = 0:  successful exit
*          < 0:  if INFO = -i, the i-th argument had an illegal value
*          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
*                has been completed, but the factor U is exactly
*                singular, and division by zero will occur if it is used
*                to solve a system of equations.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE
      PARAMETER          ( ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, IINFO, J, JB, NB
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEMM, DGETF2, DLASWP, DTRSM, XERBLA
*     ..
*     .. External Functions ..
      INTEGER            ILAENV
      EXTERNAL           ILAENV
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      IF( M.LT.0 ) THEN
         INFO = -1
      ELSE IF( N.LT.0 ) THEN
         INFO = -2
      ELSE IF( LDA.LT.MAX( 1, M ) ) THEN
         INFO = -4
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DGETRF', -INFO )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( M.EQ.0 .OR. N.EQ.0 )
     $   RETURN
*
*     Determine the block size for this environment.
*
      NB = ILAENV( 1, 'DGETRF', ' ', M, N, -1, -1 )
      IF( NB.LE.1 .OR. NB.GE.MIN( M, N ) ) THEN
*
*        Use unblocked code.
*
         CALL DGETF2( M, N, A, LDA, IPIV, INFO )
      ELSE
*
*        Use blocked code.
*
         DO 20 J = 1, MIN( M, N ), NB
            JB = MIN( MIN( M, N )-J+1, NB )
*
*           Factor diagonal and subdiagonal blocks and test for exact
*           singularity.
*
            CALL DGETF2( M-J+1, JB, A( J, J ), LDA, IPIV( J ), IINFO )
*
*           Adjust INFO and the pivot indices.
*
            IF( INFO.EQ.0 .AND. IINFO.GT.0 )
     $         INFO = IINFO + J - 1
            DO 10 I = J, MIN( M, J+JB-1 )
               IPIV( I ) = J - 1 + IPIV( I )
   10       CONTINUE
*
*           Apply interchanges to columns 1:J-1.
*
            CALL DLASWP( J-1, A, LDA, J, J+JB-1, IPIV, 1 )
*
            IF( J+JB.LE.N ) THEN
*
*              Apply interchanges to columns J+JB:N.
*
               CALL DLASWP( N-J-JB+1, A( 1, J+JB ), LDA, J, J+JB-1,
     $                      IPIV, 1 )
*
*              Compute block row of U.
*
               CALL DTRSM( 'Left', 'Lower', 'No transpose', 'Unit', JB,
     $                     N-J-JB+1, ONE, A( J, J ), LDA, A( J, J+JB ),
     $                     LDA )
               IF( J+JB.LE.M ) THEN
*
*                 Update trailing submatrix.
*
                  CALL DGEMM( 'No transpose', 'No transpose', M-J-JB+1,
     $                        N-J-JB+1, JB, -ONE, A( J+JB, J ), LDA,
     $                        A( J, J+JB ), LDA, ONE, A( J+JB, J+JB ),
     $                        LDA )
               END IF
            END IF
   20    CONTINUE
      END IF
      RETURN
*
*     End of DGETRF
*
      END
      SUBROUTINE DLASWP( N, A, LDA, K1, K2, IPIV, INCX )
*
*  -- LAPACK auxiliary routine (version 3.2) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INCX, K1, K2, LDA, N
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      DOUBLE PRECISION   A( LDA, * )
*     ..
*
*  Purpose
*  =======
*
*  DLASWP performs a series of row interchanges on the matrix A.
*  One row interchange is initiated for each of rows K1 through K2 of A.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the matrix of column dimension N to which the row
*          interchanges will be applied.
*          On exit, the permuted matrix.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.
*
*  K1      (input) INTEGER
*          The first element of IPIV for which a row interchange will
*          be done.
*
*  K2      (input) INTEGER
*          The last element of IPIV for which a row interchange will
*          be done.
*
*  IPIV    (input) INTEGER array, dimension (K2*abs(INCX))
*          The vector of pivot indices.  Only the elements in positions
*          K1 through K2 of IPIV are accessed.
*          IPIV(K) = L implies rows K and L are to be interchanged.
*
*  INCX    (input) INTEGER
*          The increment between successive values of IPIV.  If IPIV
*          is negative, the pivots are applied in reverse order.
*
*  Further Details
*  ===============
*
*  Modified by
*   R. C. Whaley, Computer Science Dept., Univ. of Tenn., Knoxville, USA
*
* =====================================================================
*
*     .. Local Scalars ..
      INTEGER            I, I1, I2, INC, IP, IX, IX0, J, K, N32
      DOUBLE PRECISION   TEMP
*     ..
*     .. Executable Statements ..
*
*     Interchange row I with row IPIV(I) for each of rows K1 through K2.
*
      IF( INCX.GT.0 ) THEN
         IX0 = K1
         I1 = K1
         I2 = K2
         INC = 1
      ELSE IF( INCX.LT.0 ) THEN
         IX0 = 1 + ( 1-K2 )*INCX
         I1 = K2
         I2 = K1
         INC = -1
      ELSE
         RETURN
      END IF
*
      N32 = ( N / 32 )*32
      IF( N32.NE.0 ) THEN
         DO 30 J = 1, N32, 32
            IX = IX0
            DO 20 I = I1, I2, INC
               IP = IPIV( IX )
               IF( IP.NE.I ) THEN
                  DO 10 K = J, J + 31
                     TEMP = A( I, K )
                     A( I, K ) = A( IP, K )
                     A( IP, K ) = TEMP
   10             CONTINUE
               END IF
               IX = IX + INCX
   20       CONTINUE
   30    CONTINUE
      END IF
      IF( N32.NE.N ) THEN
         N32 = N32 + 1
         IX = IX0
         DO 50 I = I1, I2, INC
            IP = IPIV( IX )
            IF( IP.NE.I ) THEN
               DO 40 K = N32, N
                  TEMP = A( I, K )
                  A( I, K ) = A( IP, K )
                  A( IP, K ) = TEMP
   40          CONTINUE
            END IF
            IX = IX + INCX
   50    CONTINUE
      END IF
*
      RETURN
*
*     End of DLASWP
*
      END
      DOUBLE PRECISION FUNCTION DLAMCH( CMACH )
*
*  -- LAPACK auxiliary routine (version 3.3.0) --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     Based on LAPACK DLAMCH but with Fortran 95 query functions
*     See: http://www.cs.utk.edu/~luszczek/lapack/lamch.html
*     and  http://www.netlib.org/lapack-dev/lapack-coding/program-style.html#id2537289
*     July 2010
*
*     .. Scalar Arguments ..
      CHARACTER          CMACH
*     ..
*
*  Purpose
*  =======
*
*  DLAMCH determines double precision machine parameters.
*
*  Arguments
*  =========
*
*  CMACH   (input) CHARACTER*1
*          Specifies the value to be returned by DLAMCH:
*          = 'E' or 'e',   DLAMCH := eps
*          = 'S' or 's ,   DLAMCH := sfmin
*          = 'B' or 'b',   DLAMCH := base
*          = 'P' or 'p',   DLAMCH := eps*base
*          = 'N' or 'n',   DLAMCH := t
*          = 'R' or 'r',   DLAMCH := rnd
*          = 'M' or 'm',   DLAMCH := emin
*          = 'U' or 'u',   DLAMCH := rmin
*          = 'L' or 'l',   DLAMCH := emax
*          = 'O' or 'o',   DLAMCH := rmax
*
*          where
*
*          eps   = relative machine precision
*          sfmin = safe minimum, such that 1/sfmin does not overflow
*          base  = base of the machine
*          prec  = eps*base
*          t     = number of (base) digits in the mantissa
*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise
*          emin  = minimum exponent before (gradual) underflow
*          rmin  = underflow threshold - base**(emin-1)
*          emax  = largest exponent before overflow
*          rmax  = overflow threshold  - (base**emax)*(1-eps)
*
* =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION   RND, EPS, SFMIN, SMALL, RMACH
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      EXTERNAL           LSAME
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          DIGITS, EPSILON, HUGE, MAXEXPONENT,
     $                   MINEXPONENT, RADIX, TINY
*     ..
*     .. Executable Statements ..
*
*
*     Assume rounding, not chopping. Always.
*
      RND = ONE
*
      IF( ONE.EQ.RND ) THEN
         EPS = EPSILON(ZERO) * 0.5
      ELSE
         EPS = EPSILON(ZERO)
      END IF
*
      IF( LSAME( CMACH, 'E' ) ) THEN
         RMACH = EPS
      ELSE IF( LSAME( CMACH, 'S' ) ) THEN
         SFMIN = TINY(ZERO)
         SMALL = ONE / HUGE(ZERO)
         IF( SMALL.GE.SFMIN ) THEN
*
*           Use SMALL plus a bit, to avoid the possibility of rounding
*           causing overflow when computing  1/sfmin.
*
            SFMIN = SMALL*( ONE+EPS )
         END IF
         RMACH = SFMIN
      ELSE IF( LSAME( CMACH, 'B' ) ) THEN
         RMACH = RADIX(ZERO)
      ELSE IF( LSAME( CMACH, 'P' ) ) THEN
         RMACH = EPS * RADIX(ZERO)
      ELSE IF( LSAME( CMACH, 'N' ) ) THEN
         RMACH = DIGITS(ZERO)
      ELSE IF( LSAME( CMACH, 'R' ) ) THEN
         RMACH = RND
      ELSE IF( LSAME( CMACH, 'M' ) ) THEN
         RMACH = MINEXPONENT(ZERO)
      ELSE IF( LSAME( CMACH, 'U' ) ) THEN
         RMACH = tiny(zero)
      ELSE IF( LSAME( CMACH, 'L' ) ) THEN
         RMACH = MAXEXPONENT(ZERO)
      ELSE IF( LSAME( CMACH, 'O' ) ) THEN
         RMACH = HUGE(ZERO)
      ELSE
         RMACH = ZERO
      END IF
*
      DLAMCH = RMACH
      RETURN
*
*     End of DLAMCH
*
      END
************************************************************************
*
      DOUBLE PRECISION FUNCTION DLAMC3( A, B )
*
*  -- LAPACK auxiliary routine (version 3.3.0) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2010
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION   A, B
*     ..
*
*  Purpose
*  =======
*
*  DLAMC3  is intended to force  A  and  B  to be stored prior to doing
*  the addition of  A  and  B ,  for use in situations where optimizers
*  might hold one of these in a register.
*
*  Arguments
*  =========
*
*  A       (input) DOUBLE PRECISION
*  B       (input) DOUBLE PRECISION
*          The values A and B.
*
* =====================================================================
*
*     .. Executable Statements ..
*
      DLAMC3 = A + B
*
      RETURN
*
*     End of DLAMC3
*
      END
*
************************************************************************
c$$$      INTEGER          FUNCTION IEEECK( ISPEC, ZERO, ONE )
c$$$*
c$$$*  -- LAPACK auxiliary routine (version 3.3.1) --
c$$$*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
c$$$*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
c$$$*  -- April 2011                                                      --
c$$$*
c$$$*     .. Scalar Arguments ..
c$$$      INTEGER            ISPEC
c$$$      REAL               ONE, ZERO
c$$$*     ..
c$$$*
c$$$*  Purpose
c$$$*  =======
c$$$*
c$$$*  IEEECK is called from the ILAENV to verify that Infinity and
c$$$*  possibly NaN arithmetic is safe (i.e. will not trap).
c$$$*
c$$$*  Arguments
c$$$*  =========
c$$$*
c$$$*  ISPEC   (input) INTEGER
c$$$*          Specifies whether to test just for inifinity arithmetic
c$$$*          or whether to test for infinity and NaN arithmetic.
c$$$*          = 0: Verify infinity arithmetic only.
c$$$*          = 1: Verify infinity and NaN arithmetic.
c$$$*
c$$$*  ZERO    (input) REAL
c$$$*          Must contain the value 0.0
c$$$*          This is passed to prevent the compiler from optimizing
c$$$*          away this code.
c$$$*
c$$$*  ONE     (input) REAL
c$$$*          Must contain the value 1.0
c$$$*          This is passed to prevent the compiler from optimizing
c$$$*          away this code.
c$$$*
c$$$*  RETURN VALUE:  INTEGER
c$$$*          = 0:  Arithmetic failed to produce the correct answers
c$$$*          = 1:  Arithmetic produced the correct answers
c$$$*
c$$$*  =====================================================================
c$$$*
c$$$*     .. Local Scalars ..
c$$$      REAL               NAN1, NAN2, NAN3, NAN4, NAN5, NAN6, NEGINF,
c$$$     $                   NEGZRO, NEWZRO, POSINF
c$$$*     ..
c$$$*     .. Executable Statements ..
c$$$      IEEECK = 1
c$$$*
c$$$      POSINF = ONE / ZERO
c$$$      IF( POSINF.LE.ONE ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      NEGINF = -ONE / ZERO
c$$$      IF( NEGINF.GE.ZERO ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      NEGZRO = ONE / ( NEGINF+ONE )
c$$$      IF( NEGZRO.NE.ZERO ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      NEGINF = ONE / NEGZRO
c$$$      IF( NEGINF.GE.ZERO ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      NEWZRO = NEGZRO + ZERO
c$$$      IF( NEWZRO.NE.ZERO ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      POSINF = ONE / NEWZRO
c$$$      IF( POSINF.LE.ONE ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      NEGINF = NEGINF*POSINF
c$$$      IF( NEGINF.GE.ZERO ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      POSINF = POSINF*POSINF
c$$$      IF( POSINF.LE.ONE ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$*
c$$$*
c$$$*
c$$$*     Return if we were only asked to check infinity arithmetic
c$$$*
c$$$      IF( ISPEC.EQ.0 )
c$$$     $   RETURN
c$$$*
c$$$      NAN1 = POSINF + NEGINF
c$$$*
c$$$      NAN2 = POSINF / NEGINF
c$$$*
c$$$      NAN3 = POSINF / POSINF
c$$$*
c$$$      NAN4 = POSINF*ZERO
c$$$*
c$$$      NAN5 = NEGINF*NEGZRO
c$$$*
c$$$      NAN6 = NAN5*ZERO
c$$$*
c$$$      IF( NAN1.EQ.NAN1 ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      IF( NAN2.EQ.NAN2 ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      IF( NAN3.EQ.NAN3 ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      IF( NAN4.EQ.NAN4 ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      IF( NAN5.EQ.NAN5 ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      IF( NAN6.EQ.NAN6 ) THEN
c$$$         IEEECK = 0
c$$$         RETURN
c$$$      END IF
c$$$*
c$$$      RETURN
c$$$      END
c$$$      INTEGER FUNCTION ILAENV( ISPEC, NAME, OPTS, N1, N2, N3, N4 )
c$$$*
c$$$*  -- LAPACK auxiliary routine (version 3.2.1)                        --
c$$$*
c$$$*  -- April 2009                                                      --
c$$$*
c$$$*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
c$$$*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
c$$$*
c$$$*     .. Scalar Arguments ..
c$$$      CHARACTER*( * )    NAME, OPTS
c$$$      INTEGER            ISPEC, N1, N2, N3, N4
c$$$*     ..
c$$$*
c$$$*  Purpose
c$$$*  =======
c$$$*
c$$$*  ILAENV is called from the LAPACK routines to choose problem-dependent
c$$$*  parameters for the local environment.  See ISPEC for a description of
c$$$*  the parameters.
c$$$*
c$$$*  ILAENV returns an INTEGER
c$$$*  if ILAENV >= 0: ILAENV returns the value of the parameter specified by ISPEC
c$$$*  if ILAENV < 0:  if ILAENV = -k, the k-th argument had an illegal value.
c$$$*
c$$$*  This version provides a set of parameters which should give good,
c$$$*  but not optimal, performance on many of the currently available
c$$$*  computers.  Users are encouraged to modify this subroutine to set
c$$$*  the tuning parameters for their particular machine using the option
c$$$*  and problem size information in the arguments.
c$$$*
c$$$*  This routine will not function correctly if it is converted to all
c$$$*  lower case.  Converting it to all upper case is allowed.
c$$$*
c$$$*  Arguments
c$$$*  =========
c$$$*
c$$$*  ISPEC   (input) INTEGER
c$$$*          Specifies the parameter to be returned as the value of
c$$$*          ILAENV.
c$$$*          = 1: the optimal blocksize; if this value is 1, an unblocked
c$$$*               algorithm will give the best performance.
c$$$*          = 2: the minimum block size for which the block routine
c$$$*               should be used; if the usable block size is less than
c$$$*               this value, an unblocked routine should be used.
c$$$*          = 3: the crossover point (in a block routine, for N less
c$$$*               than this value, an unblocked routine should be used)
c$$$*          = 4: the number of shifts, used in the nonsymmetric
c$$$*               eigenvalue routines (DEPRECATED)
c$$$*          = 5: the minimum column dimension for blocking to be used;
c$$$*               rectangular blocks must have dimension at least k by m,
c$$$*               where k is given by ILAENV(2,...) and m by ILAENV(5,...)
c$$$*          = 6: the crossover point for the SVD (when reducing an m by n
c$$$*               matrix to bidiagonal form, if max(m,n)/min(m,n) exceeds
c$$$*               this value, a QR factorization is used first to reduce
c$$$*               the matrix to a triangular form.)
c$$$*          = 7: the number of processors
c$$$*          = 8: the crossover point for the multishift QR method
c$$$*               for nonsymmetric eigenvalue problems (DEPRECATED)
c$$$*          = 9: maximum size of the subproblems at the bottom of the
c$$$*               computation tree in the divide-and-conquer algorithm
c$$$*               (used by xGELSD and xGESDD)
c$$$*          =10: ieee NaN arithmetic can be trusted not to trap
c$$$*          =11: infinity arithmetic can be trusted not to trap
c$$$*          12 <= ISPEC <= 16:
c$$$*               xHSEQR or one of its subroutines,
c$$$*               see IPARMQ for detailed explanation
c$$$*
c$$$*  NAME    (input) CHARACTER*(*)
c$$$*          The name of the calling subroutine, in either upper case or
c$$$*          lower case.
c$$$*
c$$$*  OPTS    (input) CHARACTER*(*)
c$$$*          The character options to the subroutine NAME, concatenated
c$$$*          into a single character string.  For example, UPLO = 'U',
c$$$*          TRANS = 'T', and DIAG = 'N' for a triangular routine would
c$$$*          be specified as OPTS = 'UTN'.
c$$$*
c$$$*  N1      (input) INTEGER
c$$$*  N2      (input) INTEGER
c$$$*  N3      (input) INTEGER
c$$$*  N4      (input) INTEGER
c$$$*          Problem dimensions for the subroutine NAME; these may not all
c$$$*          be required.
c$$$*
c$$$*  Further Details
c$$$*  ===============
c$$$*
c$$$*  The following conventions have been used when calling ILAENV from the
c$$$*  LAPACK routines:
c$$$*  1)  OPTS is a concatenation of all of the character options to
c$$$*      subroutine NAME, in the same order that they appear in the
c$$$*      argument list for NAME, even if they are not used in determining
c$$$*      the value of the parameter specified by ISPEC.
c$$$*  2)  The problem dimensions N1, N2, N3, N4 are specified in the order
c$$$*      that they appear in the argument list for NAME.  N1 is used
c$$$*      first, N2 second, and so on, and unused problem dimensions are
c$$$*      passed a value of -1.
c$$$*  3)  The parameter value returned by ILAENV is checked for validity in
c$$$*      the calling subroutine.  For example, ILAENV is used to retrieve
c$$$*      the optimal blocksize for STRTRI as follows:
c$$$*
c$$$*      NB = ILAENV( 1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1 )
c$$$*      IF( NB.LE.1 ) NB = MAX( 1, N )
c$$$*
c$$$*  =====================================================================
c$$$*
c$$$*     .. Local Scalars ..
c$$$      INTEGER            I, IC, IZ, NB, NBMIN, NX
c$$$      LOGICAL            CNAME, SNAME
c$$$      CHARACTER          C1*1, C2*2, C4*2, C3*3, SUBNAM*6
c$$$*     ..
c$$$*     .. Intrinsic Functions ..
c$$$      INTRINSIC          CHAR, ICHAR, INT, MIN, REAL
c$$$*     ..
c$$$*     .. External Functions ..
c$$$      INTEGER            IEEECK, IPARMQ
c$$$      EXTERNAL           IEEECK, IPARMQ
c$$$*     ..
c$$$*     .. Executable Statements ..
c$$$*
c$$$      GO TO ( 10, 10, 10, 80, 90, 100, 110, 120,
c$$$     $        130, 140, 150, 160, 160, 160, 160, 160 )ISPEC
c$$$*
c$$$*     Invalid value for ISPEC
c$$$*
c$$$      ILAENV = -1
c$$$      RETURN
c$$$*
c$$$   10 CONTINUE
c$$$*
c$$$*     Convert NAME to upper case if the first character is lower case.
c$$$*
c$$$      ILAENV = 1
c$$$      SUBNAM = NAME
c$$$      IC = ICHAR( SUBNAM( 1: 1 ) )
c$$$      IZ = ICHAR( 'Z' )
c$$$      IF( IZ.EQ.90 .OR. IZ.EQ.122 ) THEN
c$$$*
c$$$*        ASCII character set
c$$$*
c$$$         IF( IC.GE.97 .AND. IC.LE.122 ) THEN
c$$$            SUBNAM( 1: 1 ) = CHAR( IC-32 )
c$$$            DO 20 I = 2, 6
c$$$               IC = ICHAR( SUBNAM( I: I ) )
c$$$               IF( IC.GE.97 .AND. IC.LE.122 )
c$$$     $            SUBNAM( I: I ) = CHAR( IC-32 )
c$$$   20       CONTINUE
c$$$         END IF
c$$$*
c$$$      ELSE IF( IZ.EQ.233 .OR. IZ.EQ.169 ) THEN
c$$$*
c$$$*        EBCDIC character set
c$$$*
c$$$         IF( ( IC.GE.129 .AND. IC.LE.137 ) .OR.
c$$$     $       ( IC.GE.145 .AND. IC.LE.153 ) .OR.
c$$$     $       ( IC.GE.162 .AND. IC.LE.169 ) ) THEN
c$$$            SUBNAM( 1: 1 ) = CHAR( IC+64 )
c$$$            DO 30 I = 2, 6
c$$$               IC = ICHAR( SUBNAM( I: I ) )
c$$$               IF( ( IC.GE.129 .AND. IC.LE.137 ) .OR.
c$$$     $             ( IC.GE.145 .AND. IC.LE.153 ) .OR.
c$$$     $             ( IC.GE.162 .AND. IC.LE.169 ) )SUBNAM( I:
c$$$     $             I ) = CHAR( IC+64 )
c$$$   30       CONTINUE
c$$$         END IF
c$$$*
c$$$      ELSE IF( IZ.EQ.218 .OR. IZ.EQ.250 ) THEN
c$$$*
c$$$*        Prime machines:  ASCII+128
c$$$*
c$$$         IF( IC.GE.225 .AND. IC.LE.250 ) THEN
c$$$            SUBNAM( 1: 1 ) = CHAR( IC-32 )
c$$$            DO 40 I = 2, 6
c$$$               IC = ICHAR( SUBNAM( I: I ) )
c$$$               IF( IC.GE.225 .AND. IC.LE.250 )
c$$$     $            SUBNAM( I: I ) = CHAR( IC-32 )
c$$$   40       CONTINUE
c$$$         END IF
c$$$      END IF
c$$$*
c$$$      C1 = SUBNAM( 1: 1 )
c$$$      SNAME = C1.EQ.'S' .OR. C1.EQ.'D'
c$$$      CNAME = C1.EQ.'C' .OR. C1.EQ.'Z'
c$$$      IF( .NOT.( CNAME .OR. SNAME ) )
c$$$     $   RETURN
c$$$      C2 = SUBNAM( 2: 3 )
c$$$      C3 = SUBNAM( 4: 6 )
c$$$      C4 = C3( 2: 3 )
c$$$*
c$$$      GO TO ( 50, 60, 70 )ISPEC
c$$$*
c$$$   50 CONTINUE
c$$$*
c$$$*     ISPEC = 1:  block size
c$$$*
c$$$*     In these examples, separate code is provided for setting NB for
c$$$*     real and complex.  We assume that NB will take the same value in
c$$$*     single or double precision.
c$$$*
c$$$      NB = 1
c$$$*
c$$$      IF( C2.EQ.'GE' ) THEN
c$$$         IF( C3.EQ.'TRF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 64
c$$$            ELSE
c$$$               NB = 64
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'QRF' .OR. C3.EQ.'RQF' .OR. C3.EQ.'LQF' .OR.
c$$$     $            C3.EQ.'QLF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 32
c$$$            ELSE
c$$$               NB = 32
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'HRD' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 32
c$$$            ELSE
c$$$               NB = 32
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'BRD' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 32
c$$$            ELSE
c$$$               NB = 32
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'TRI' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 64
c$$$            ELSE
c$$$               NB = 64
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'PO' ) THEN
c$$$         IF( C3.EQ.'TRF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 64
c$$$            ELSE
c$$$               NB = 64
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'SY' ) THEN
c$$$         IF( C3.EQ.'TRF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 64
c$$$            ELSE
c$$$               NB = 64
c$$$            END IF
c$$$         ELSE IF( SNAME .AND. C3.EQ.'TRD' ) THEN
c$$$            NB = 32
c$$$         ELSE IF( SNAME .AND. C3.EQ.'GST' ) THEN
c$$$            NB = 64
c$$$         END IF
c$$$      ELSE IF( CNAME .AND. C2.EQ.'HE' ) THEN
c$$$         IF( C3.EQ.'TRF' ) THEN
c$$$            NB = 64
c$$$         ELSE IF( C3.EQ.'TRD' ) THEN
c$$$            NB = 32
c$$$         ELSE IF( C3.EQ.'GST' ) THEN
c$$$            NB = 64
c$$$         END IF
c$$$      ELSE IF( SNAME .AND. C2.EQ.'OR' ) THEN
c$$$         IF( C3( 1: 1 ).EQ.'G' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NB = 32
c$$$            END IF
c$$$         ELSE IF( C3( 1: 1 ).EQ.'M' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NB = 32
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( CNAME .AND. C2.EQ.'UN' ) THEN
c$$$         IF( C3( 1: 1 ).EQ.'G' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NB = 32
c$$$            END IF
c$$$         ELSE IF( C3( 1: 1 ).EQ.'M' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NB = 32
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'GB' ) THEN
c$$$         IF( C3.EQ.'TRF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               IF( N4.LE.64 ) THEN
c$$$                  NB = 1
c$$$               ELSE
c$$$                  NB = 32
c$$$               END IF
c$$$            ELSE
c$$$               IF( N4.LE.64 ) THEN
c$$$                  NB = 1
c$$$               ELSE
c$$$                  NB = 32
c$$$               END IF
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'PB' ) THEN
c$$$         IF( C3.EQ.'TRF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               IF( N2.LE.64 ) THEN
c$$$                  NB = 1
c$$$               ELSE
c$$$                  NB = 32
c$$$               END IF
c$$$            ELSE
c$$$               IF( N2.LE.64 ) THEN
c$$$                  NB = 1
c$$$               ELSE
c$$$                  NB = 32
c$$$               END IF
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'TR' ) THEN
c$$$         IF( C3.EQ.'TRI' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 64
c$$$            ELSE
c$$$               NB = 64
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'LA' ) THEN
c$$$         IF( C3.EQ.'UUM' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NB = 64
c$$$            ELSE
c$$$               NB = 64
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( SNAME .AND. C2.EQ.'ST' ) THEN
c$$$         IF( C3.EQ.'EBZ' ) THEN
c$$$            NB = 1
c$$$         END IF
c$$$      END IF
c$$$      ILAENV = NB
c$$$      RETURN
c$$$*
c$$$   60 CONTINUE
c$$$*
c$$$*     ISPEC = 2:  minimum block size
c$$$*
c$$$      NBMIN = 2
c$$$      IF( C2.EQ.'GE' ) THEN
c$$$         IF( C3.EQ.'QRF' .OR. C3.EQ.'RQF' .OR. C3.EQ.'LQF' .OR. C3.EQ.
c$$$     $       'QLF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NBMIN = 2
c$$$            ELSE
c$$$               NBMIN = 2
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'HRD' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NBMIN = 2
c$$$            ELSE
c$$$               NBMIN = 2
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'BRD' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NBMIN = 2
c$$$            ELSE
c$$$               NBMIN = 2
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'TRI' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NBMIN = 2
c$$$            ELSE
c$$$               NBMIN = 2
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'SY' ) THEN
c$$$         IF( C3.EQ.'TRF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NBMIN = 8
c$$$            ELSE
c$$$               NBMIN = 8
c$$$            END IF
c$$$         ELSE IF( SNAME .AND. C3.EQ.'TRD' ) THEN
c$$$            NBMIN = 2
c$$$         END IF
c$$$      ELSE IF( CNAME .AND. C2.EQ.'HE' ) THEN
c$$$         IF( C3.EQ.'TRD' ) THEN
c$$$            NBMIN = 2
c$$$         END IF
c$$$      ELSE IF( SNAME .AND. C2.EQ.'OR' ) THEN
c$$$         IF( C3( 1: 1 ).EQ.'G' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NBMIN = 2
c$$$            END IF
c$$$         ELSE IF( C3( 1: 1 ).EQ.'M' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NBMIN = 2
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( CNAME .AND. C2.EQ.'UN' ) THEN
c$$$         IF( C3( 1: 1 ).EQ.'G' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NBMIN = 2
c$$$            END IF
c$$$         ELSE IF( C3( 1: 1 ).EQ.'M' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NBMIN = 2
c$$$            END IF
c$$$         END IF
c$$$      END IF
c$$$      ILAENV = NBMIN
c$$$      RETURN
c$$$*
c$$$   70 CONTINUE
c$$$*
c$$$*     ISPEC = 3:  crossover point
c$$$*
c$$$      NX = 0
c$$$      IF( C2.EQ.'GE' ) THEN
c$$$         IF( C3.EQ.'QRF' .OR. C3.EQ.'RQF' .OR. C3.EQ.'LQF' .OR. C3.EQ.
c$$$     $       'QLF' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NX = 128
c$$$            ELSE
c$$$               NX = 128
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'HRD' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NX = 128
c$$$            ELSE
c$$$               NX = 128
c$$$            END IF
c$$$         ELSE IF( C3.EQ.'BRD' ) THEN
c$$$            IF( SNAME ) THEN
c$$$               NX = 128
c$$$            ELSE
c$$$               NX = 128
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( C2.EQ.'SY' ) THEN
c$$$         IF( SNAME .AND. C3.EQ.'TRD' ) THEN
c$$$            NX = 32
c$$$         END IF
c$$$      ELSE IF( CNAME .AND. C2.EQ.'HE' ) THEN
c$$$         IF( C3.EQ.'TRD' ) THEN
c$$$            NX = 32
c$$$         END IF
c$$$      ELSE IF( SNAME .AND. C2.EQ.'OR' ) THEN
c$$$         IF( C3( 1: 1 ).EQ.'G' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NX = 128
c$$$            END IF
c$$$         END IF
c$$$      ELSE IF( CNAME .AND. C2.EQ.'UN' ) THEN
c$$$         IF( C3( 1: 1 ).EQ.'G' ) THEN
c$$$            IF( C4.EQ.'QR' .OR. C4.EQ.'RQ' .OR. C4.EQ.'LQ' .OR. C4.EQ.
c$$$     $          'QL' .OR. C4.EQ.'HR' .OR. C4.EQ.'TR' .OR. C4.EQ.'BR' )
c$$$     $           THEN
c$$$               NX = 128
c$$$            END IF
c$$$         END IF
c$$$      END IF
c$$$      ILAENV = NX
c$$$      RETURN
c$$$*
c$$$   80 CONTINUE
c$$$*
c$$$*     ISPEC = 4:  number of shifts (used by xHSEQR)
c$$$*
c$$$      ILAENV = 6
c$$$      RETURN
c$$$*
c$$$   90 CONTINUE
c$$$*
c$$$*     ISPEC = 5:  minimum column dimension (not used)
c$$$*
c$$$      ILAENV = 2
c$$$      RETURN
c$$$*
c$$$  100 CONTINUE
c$$$*
c$$$*     ISPEC = 6:  crossover point for SVD (used by xGELSS and xGESVD)
c$$$*
c$$$      ILAENV = INT( REAL( MIN( N1, N2 ) )*1.6E0 )
c$$$      RETURN
c$$$*
c$$$  110 CONTINUE
c$$$*
c$$$*     ISPEC = 7:  number of processors (not used)
c$$$*
c$$$      ILAENV = 1
c$$$      RETURN
c$$$*
c$$$  120 CONTINUE
c$$$*
c$$$*     ISPEC = 8:  crossover point for multishift (used by xHSEQR)
c$$$*
c$$$      ILAENV = 50
c$$$      RETURN
c$$$*
c$$$  130 CONTINUE
c$$$*
c$$$*     ISPEC = 9:  maximum size of the subproblems at the bottom of the
c$$$*                 computation tree in the divide-and-conquer algorithm
c$$$*                 (used by xGELSD and xGESDD)
c$$$*
c$$$      ILAENV = 25
c$$$      RETURN
c$$$*
c$$$  140 CONTINUE
c$$$*
c$$$*     ISPEC = 10: ieee NaN arithmetic can be trusted not to trap
c$$$*
c$$$*     ILAENV = 0
c$$$      ILAENV = 1
c$$$      IF( ILAENV.EQ.1 ) THEN
c$$$         ILAENV = IEEECK( 1, 0.0, 1.0 )
c$$$      END IF
c$$$      RETURN
c$$$*
c$$$  150 CONTINUE
c$$$*
c$$$*     ISPEC = 11: infinity arithmetic can be trusted not to trap
c$$$*
c$$$*     ILAENV = 0
c$$$      ILAENV = 1
c$$$      IF( ILAENV.EQ.1 ) THEN
c$$$         ILAENV = IEEECK( 0, 0.0, 1.0 )
c$$$      END IF
c$$$      RETURN
c$$$*
c$$$  160 CONTINUE
c$$$*
c$$$*     12 <= ISPEC <= 16: xHSEQR or one of its subroutines. 
c$$$*
c$$$      ILAENV = IPARMQ( ISPEC, NAME, OPTS, N1, N2, N3, N4 )
c$$$      RETURN
c$$$*
c$$$*     End of ILAENV
c$$$*
c$$$      END
c$$$      INTEGER FUNCTION IPARMQ( ISPEC, NAME, OPTS, N, ILO, IHI, LWORK )
c$$$*
c$$$*  -- LAPACK auxiliary routine (version 3.2) --
c$$$*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
c$$$*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
c$$$*     November 2006
c$$$*     
c$$$*     .. Scalar Arguments ..
c$$$      INTEGER            IHI, ILO, ISPEC, LWORK, N
c$$$      CHARACTER          NAME*( * ), OPTS*( * )
c$$$*
c$$$*  Purpose
c$$$*  =======
c$$$*
c$$$*       This program sets problem and machine dependent parameters
c$$$*       useful for xHSEQR and its subroutines. It is called whenever 
c$$$*       ILAENV is called with 12 <= ISPEC <= 16
c$$$*
c$$$*  Arguments
c$$$*  =========
c$$$*
c$$$*       ISPEC  (input) integer scalar
c$$$*              ISPEC specifies which tunable parameter IPARMQ should
c$$$*              return.
c$$$*
c$$$*              ISPEC=12: (INMIN)  Matrices of order nmin or less
c$$$*                        are sent directly to xLAHQR, the implicit
c$$$*                        double shift QR algorithm.  NMIN must be
c$$$*                        at least 11.
c$$$*
c$$$*              ISPEC=13: (INWIN)  Size of the deflation window.
c$$$*                        This is best set greater than or equal to
c$$$*                        the number of simultaneous shifts NS.
c$$$*                        Larger matrices benefit from larger deflation
c$$$*                        windows.
c$$$*
c$$$*              ISPEC=14: (INIBL) Determines when to stop nibbling and
c$$$*                        invest in an (expensive) multi-shift QR sweep.
c$$$*                        If the aggressive early deflation subroutine
c$$$*                        finds LD converged eigenvalues from an order
c$$$*                        NW deflation window and LD.GT.(NW*NIBBLE)/100,
c$$$*                        then the next QR sweep is skipped and early
c$$$*                        deflation is applied immediately to the
c$$$*                        remaining active diagonal block.  Setting
c$$$*                        IPARMQ(ISPEC=14) = 0 causes TTQRE to skip a
c$$$*                        multi-shift QR sweep whenever early deflation
c$$$*                        finds a converged eigenvalue.  Setting
c$$$*                        IPARMQ(ISPEC=14) greater than or equal to 100
c$$$*                        prevents TTQRE from skipping a multi-shift
c$$$*                        QR sweep.
c$$$*
c$$$*              ISPEC=15: (NSHFTS) The number of simultaneous shifts in
c$$$*                        a multi-shift QR iteration.
c$$$*
c$$$*              ISPEC=16: (IACC22) IPARMQ is set to 0, 1 or 2 with the
c$$$*                        following meanings.
c$$$*                        0:  During the multi-shift QR sweep,
c$$$*                            xLAQR5 does not accumulate reflections and
c$$$*                            does not use matrix-matrix multiply to
c$$$*                            update the far-from-diagonal matrix
c$$$*                            entries.
c$$$*                        1:  During the multi-shift QR sweep,
c$$$*                            xLAQR5 and/or xLAQRaccumulates reflections and uses
c$$$*                            matrix-matrix multiply to update the
c$$$*                            far-from-diagonal matrix entries.
c$$$*                        2:  During the multi-shift QR sweep.
c$$$*                            xLAQR5 accumulates reflections and takes
c$$$*                            advantage of 2-by-2 block structure during
c$$$*                            matrix-matrix multiplies.
c$$$*                        (If xTRMM is slower than xGEMM, then
c$$$*                        IPARMQ(ISPEC=16)=1 may be more efficient than
c$$$*                        IPARMQ(ISPEC=16)=2 despite the greater level of
c$$$*                        arithmetic work implied by the latter choice.)
c$$$*
c$$$*       NAME    (input) character string
c$$$*               Name of the calling subroutine
c$$$*
c$$$*       OPTS    (input) character string
c$$$*               This is a concatenation of the string arguments to
c$$$*               TTQRE.
c$$$*
c$$$*       N       (input) integer scalar
c$$$*               N is the order of the Hessenberg matrix H.
c$$$*
c$$$*       ILO     (input) INTEGER
c$$$*       IHI     (input) INTEGER
c$$$*               It is assumed that H is already upper triangular
c$$$*               in rows and columns 1:ILO-1 and IHI+1:N.
c$$$*
c$$$*       LWORK   (input) integer scalar
c$$$*               The amount of workspace available.
c$$$*
c$$$*  Further Details
c$$$*  ===============
c$$$*
c$$$*       Little is known about how best to choose these parameters.
c$$$*       It is possible to use different values of the parameters
c$$$*       for each of CHSEQR, DHSEQR, SHSEQR and ZHSEQR.
c$$$*
c$$$*       It is probably best to choose different parameters for
c$$$*       different matrices and different parameters at different
c$$$*       times during the iteration, but this has not been
c$$$*       implemented --- yet.
c$$$*
c$$$*
c$$$*       The best choices of most of the parameters depend
c$$$*       in an ill-understood way on the relative execution
c$$$*       rate of xLAQR3 and xLAQR5 and on the nature of each
c$$$*       particular eigenvalue problem.  Experiment may be the
c$$$*       only practical way to determine which choices are most
c$$$*       effective.
c$$$*
c$$$*       Following is a list of default values supplied by IPARMQ.
c$$$*       These defaults may be adjusted in order to attain better
c$$$*       performance in any particular computational environment.
c$$$*
c$$$*       IPARMQ(ISPEC=12) The xLAHQR vs xLAQR0 crossover point.
c$$$*                        Default: 75. (Must be at least 11.)
c$$$*
c$$$*       IPARMQ(ISPEC=13) Recommended deflation window size.
c$$$*                        This depends on ILO, IHI and NS, the
c$$$*                        number of simultaneous shifts returned
c$$$*                        by IPARMQ(ISPEC=15).  The default for
c$$$*                        (IHI-ILO+1).LE.500 is NS.  The default
c$$$*                        for (IHI-ILO+1).GT.500 is 3*NS/2.
c$$$*
c$$$*       IPARMQ(ISPEC=14) Nibble crossover point.  Default: 14.
c$$$*
c$$$*       IPARMQ(ISPEC=15) Number of simultaneous shifts, NS.
c$$$*                        a multi-shift QR iteration.
c$$$*
c$$$*                        If IHI-ILO+1 is ...
c$$$*
c$$$*                        greater than      ...but less    ... the
c$$$*                        or equal to ...      than        default is
c$$$*
c$$$*                                0               30       NS =   2+
c$$$*                               30               60       NS =   4+
c$$$*                               60              150       NS =  10
c$$$*                              150              590       NS =  **
c$$$*                              590             3000       NS =  64
c$$$*                             3000             6000       NS = 128
c$$$*                             6000             infinity   NS = 256
c$$$*
c$$$*                    (+)  By default matrices of this order are
c$$$*                         passed to the implicit double shift routine
c$$$*                         xLAHQR.  See IPARMQ(ISPEC=12) above.   These
c$$$*                         values of NS are used only in case of a rare
c$$$*                         xLAHQR failure.
c$$$*
c$$$*                    (**) The asterisks (**) indicate an ad-hoc
c$$$*                         function increasing from 10 to 64.
c$$$*
c$$$*       IPARMQ(ISPEC=16) Select structured matrix multiply.
c$$$*                        (See ISPEC=16 above for details.)
c$$$*                        Default: 3.
c$$$*
c$$$*     ================================================================
c$$$*     .. Parameters ..
c$$$      INTEGER            INMIN, INWIN, INIBL, ISHFTS, IACC22
c$$$      PARAMETER          ( INMIN = 12, INWIN = 13, INIBL = 14,
c$$$     $                   ISHFTS = 15, IACC22 = 16 )
c$$$      INTEGER            NMIN, K22MIN, KACMIN, NIBBLE, KNWSWP
c$$$      PARAMETER          ( NMIN = 75, K22MIN = 14, KACMIN = 14,
c$$$     $                   NIBBLE = 14, KNWSWP = 500 )
c$$$      REAL               TWO
c$$$      PARAMETER          ( TWO = 2.0 )
c$$$*     ..
c$$$*     .. Local Scalars ..
c$$$      INTEGER            NH, NS
c$$$*     ..
c$$$*     .. Intrinsic Functions ..
c$$$      INTRINSIC          LOG, MAX, MOD, NINT, REAL
c$$$*     ..
c$$$*     .. Executable Statements ..
c$$$      IF( ( ISPEC.EQ.ISHFTS ) .OR. ( ISPEC.EQ.INWIN ) .OR.
c$$$     $    ( ISPEC.EQ.IACC22 ) ) THEN
c$$$*
c$$$*        ==== Set the number simultaneous shifts ====
c$$$*
c$$$         NH = IHI - ILO + 1
c$$$         NS = 2
c$$$         IF( NH.GE.30 )
c$$$     $      NS = 4
c$$$         IF( NH.GE.60 )
c$$$     $      NS = 10
c$$$         IF( NH.GE.150 )
c$$$     $      NS = MAX( 10, NH / NINT( LOG( REAL( NH ) ) / LOG( TWO ) ) )
c$$$         IF( NH.GE.590 )
c$$$     $      NS = 64
c$$$         IF( NH.GE.3000 )
c$$$     $      NS = 128
c$$$         IF( NH.GE.6000 )
c$$$     $      NS = 256
c$$$         NS = MAX( 2, NS-MOD( NS, 2 ) )
c$$$      END IF
c$$$*
c$$$      IF( ISPEC.EQ.INMIN ) THEN
c$$$*
c$$$*
c$$$*        ===== Matrices of order smaller than NMIN get sent
c$$$*        .     to xLAHQR, the classic double shift algorithm.
c$$$*        .     This must be at least 11. ====
c$$$*
c$$$         IPARMQ = NMIN
c$$$*
c$$$      ELSE IF( ISPEC.EQ.INIBL ) THEN
c$$$*
c$$$*        ==== INIBL: skip a multi-shift qr iteration and
c$$$*        .    whenever aggressive early deflation finds
c$$$*        .    at least (NIBBLE*(window size)/100) deflations. ====
c$$$*
c$$$         IPARMQ = NIBBLE
c$$$*
c$$$      ELSE IF( ISPEC.EQ.ISHFTS ) THEN
c$$$*
c$$$*        ==== NSHFTS: The number of simultaneous shifts =====
c$$$*
c$$$         IPARMQ = NS
c$$$*
c$$$      ELSE IF( ISPEC.EQ.INWIN ) THEN
c$$$*
c$$$*        ==== NW: deflation window size.  ====
c$$$*
c$$$         IF( NH.LE.KNWSWP ) THEN
c$$$            IPARMQ = NS
c$$$         ELSE
c$$$            IPARMQ = 3*NS / 2
c$$$         END IF
c$$$*
c$$$      ELSE IF( ISPEC.EQ.IACC22 ) THEN
c$$$*
c$$$*        ==== IACC22: Whether to accumulate reflections
c$$$*        .     before updating the far-from-diagonal elements
c$$$*        .     and whether to use 2-by-2 block structure while
c$$$*        .     doing it.  A small amount of work could be saved
c$$$*        .     by making this choice dependent also upon the
c$$$*        .     NH=IHI-ILO+1.
c$$$*
c$$$         IPARMQ = 0
c$$$         IF( NS.GE.KACMIN )
c$$$     $      IPARMQ = 1
c$$$         IF( NS.GE.K22MIN )
c$$$     $      IPARMQ = 2
c$$$*
c$$$      ELSE
c$$$*        ===== invalid value of ispec =====
c$$$         IPARMQ = -1
c$$$*
c$$$      END IF
c$$$*
c$$$*     ==== End of IPARMQ ====
c$$$*
c$$$      END
c$$$      LOGICAL          FUNCTION LSAME( CA, CB )
c$$$*
c$$$*  -- LAPACK auxiliary routine (version 3.2) --
c$$$*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
c$$$*     November 2006
c$$$*
c$$$*     .. Scalar Arguments ..
c$$$      CHARACTER          CA, CB
c$$$*     ..
c$$$*
c$$$*  Purpose
c$$$*  =======
c$$$*
c$$$*  LSAME returns .TRUE. if CA is the same letter as CB regardless of
c$$$*  case.
c$$$*
c$$$*  Arguments
c$$$*  =========
c$$$*
c$$$*  CA      (input) CHARACTER*1
c$$$*  CB      (input) CHARACTER*1
c$$$*          CA and CB specify the single characters to be compared.
c$$$*
c$$$* =====================================================================
c$$$*
c$$$*     .. Intrinsic Functions ..
c$$$      INTRINSIC          ICHAR
c$$$*     ..
c$$$*     .. Local Scalars ..
c$$$      INTEGER            INTA, INTB, ZCODE
c$$$*     ..
c$$$*     .. Executable Statements ..
c$$$*
c$$$*     Test if the characters are equal
c$$$*
c$$$      LSAME = CA.EQ.CB
c$$$      IF( LSAME )
c$$$     $   RETURN
c$$$*
c$$$*     Now test for equivalence if both characters are alphabetic.
c$$$*
c$$$      ZCODE = ICHAR( 'Z' )
c$$$*
c$$$*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime
c$$$*     machines, on which ICHAR returns a value with bit 8 set.
c$$$*     ICHAR('A') on Prime machines returns 193 which is the same as
c$$$*     ICHAR('A') on an EBCDIC machine.
c$$$*
c$$$      INTA = ICHAR( CA )
c$$$      INTB = ICHAR( CB )
c$$$*
c$$$      IF( ZCODE.EQ.90 .OR. ZCODE.EQ.122 ) THEN
c$$$*
c$$$*        ASCII is assumed - ZCODE is the ASCII code of either lower or
c$$$*        upper case 'Z'.
c$$$*
c$$$         IF( INTA.GE.97 .AND. INTA.LE.122 ) INTA = INTA - 32
c$$$         IF( INTB.GE.97 .AND. INTB.LE.122 ) INTB = INTB - 32
c$$$*
c$$$      ELSE IF( ZCODE.EQ.233 .OR. ZCODE.EQ.169 ) THEN
c$$$*
c$$$*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
c$$$*        upper case 'Z'.
c$$$*
c$$$         IF( INTA.GE.129 .AND. INTA.LE.137 .OR.
c$$$     $       INTA.GE.145 .AND. INTA.LE.153 .OR.
c$$$     $       INTA.GE.162 .AND. INTA.LE.169 ) INTA = INTA + 64
c$$$         IF( INTB.GE.129 .AND. INTB.LE.137 .OR.
c$$$     $       INTB.GE.145 .AND. INTB.LE.153 .OR.
c$$$     $       INTB.GE.162 .AND. INTB.LE.169 ) INTB = INTB + 64
c$$$*
c$$$      ELSE IF( ZCODE.EQ.218 .OR. ZCODE.EQ.250 ) THEN
c$$$*
c$$$*        ASCII is assumed, on Prime machines - ZCODE is the ASCII code
c$$$*        plus 128 of either lower or upper case 'Z'.
c$$$*
c$$$         IF( INTA.GE.225 .AND. INTA.LE.250 ) INTA = INTA - 32
c$$$         IF( INTB.GE.225 .AND. INTB.LE.250 ) INTB = INTB - 32
c$$$      END IF
c$$$      LSAME = INTA.EQ.INTB
c$$$*
c$$$*     RETURN
c$$$*
c$$$*     End of LSAME
c$$$*
c$$$      END
c$$$      SUBROUTINE XERBLA( SRNAME, INFO )
c$$$*
c$$$*  -- LAPACK auxiliary routine (version 3.2) --
c$$$*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
c$$$*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
c$$$*     November 2006
c$$$*
c$$$*     .. Scalar Arguments ..
c$$$      CHARACTER*(*)      SRNAME
c$$$      INTEGER            INFO
c$$$*     ..
c$$$*
c$$$*  Purpose
c$$$*  =======
c$$$*
c$$$*  XERBLA  is an error handler for the LAPACK routines.
c$$$*  It is called by an LAPACK routine if an input parameter has an
c$$$*  invalid value.  A message is printed and execution stops.
c$$$*
c$$$*  Installers may consider modifying the STOP statement in order to
c$$$*  call system-specific exception-handling facilities.
c$$$*
c$$$*  Arguments
c$$$*  =========
c$$$*
c$$$*  SRNAME  (input) CHARACTER*(*)
c$$$*          The name of the routine which called XERBLA.
c$$$*
c$$$*  INFO    (input) INTEGER
c$$$*          The position of the invalid parameter in the parameter list
c$$$*          of the calling routine.
c$$$*
c$$$* =====================================================================
c$$$*
c$$$*     .. Intrinsic Functions ..
c$$$      INTRINSIC          LEN_TRIM
c$$$*     ..
c$$$*     .. Executable Statements ..
c$$$*
c$$$      WRITE( *, FMT = 9999 )SRNAME( 1:LEN_TRIM( SRNAME ) ), INFO
c$$$*
c$$$      STOP
c$$$*
c$$$ 9999 FORMAT( ' ** On entry to ', A, ' parameter number ', I2, ' had ',
c$$$     $      'an illegal value' )
c$$$*
c$$$*     End of XERBLA
c$$$*
c$$$      END
