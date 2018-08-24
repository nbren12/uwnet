	real function cape(n,pres,temp,qv)
C
c   based on the program calcsound from Kerry Emanuel web site.
C   ***   This program accepts input from the sounding contained   ***
C   ***      in the file <<sounding>> and calculates various       ***
C   ***       properties of samples of air raised or lowered       ***
C   ***                   to different levels.                     ***
C   ***      Output is tabulated in the file <<calcsound.out>>     ***
C   ***                 and in NCAR graphics files                 ***
C
	real pres(1), temp(1), qv(1)
c       units mb, K, g/kg
	integer n

	PARAMETER (NA=2000)
	REAL T(NA), P(NA), R(NA), TLR(1,NA), TLP(1,NA)
	REAL EV(NA), ES(NA), TVRDIF(1,NA), TVPDIF(1,NA)
	REAL TLVR(1,NA), TLVP(1,NA), LW(1,NA), TVD(NA)
	REAL CAPER(NA), CAPEP(NA), DCAPE(NA), PAR(NA)
	REAL PAP(NA), NAR(NA), NAP(NA)
C

	do i=1,n
	  p(i) = pres(i)
	  t(i) = temp(i) - 273.15
	  r(i) = qv(i)
	end do


C   ***   ASSIGN VALUES OF THERMODYNAMIC CONSTANTS     ***
C
        CPD=1005.7
	CPV=1870.0
        CL=4190.0
        CPVMCL=2320.0
        RV=461.5
        RD=287.04
        EPS=RD/RV
        ALV0=2.501E6
C
C   ***   Read in the sounding from the file <<sounding>>          ***
c
	DO 20 I=1,N
	 R(I)=R(I)*0.001
	 EV(I)=R(I)*P(I)/(EPS+R(I))
	 ES(I)=6.112*EXP(17.67*T(I)/(243.5+T(I)))
	 T(I)=T(I)+273.15
   20	CONTINUE
C
C   ***  Begin outer loop, which cycles through parcel origin levels I ***
C  
	DO 500 I=1,1 !N
C
C   ***  Define various conserved parcel quantities: reversible   ***
C   ***        entropy, S, pseudo-adiabatic entropy, SP,          *** 
C   ***                   and enthalpy, AH                        ***
C
	RS=EPS*ES(I)/(P(I)-ES(I))
	ALV=ALV0-CPVMCL*(T(I)-273.15)
	EM=MAX(EV(I),1.0E-6) 
	S=(CPD+R(I)*CL)*LOG(T(I))-RD*LOG(P(I)-EV(I))+
     1    ALV*R(I)/T(I)-R(I)*RV*LOG(EM/ES(I))
	SP=CPD*LOG(T(I))-RD*LOG(P(I)-EV(I))+
     1    ALV*R(I)/T(I)-R(I)*RV*LOG(EM/ES(I))
	AH=(CPD+R(I)*CL)*T(I)+ALV*R(I)
C      
C   ***  Find the temperature and mixing ratio of the parcel at   ***
C   ***    level I saturated by a wet bulb process                ***
C
	SLOPE=CPD+ALV*ALV*RS/(RV*T(I)*T(I))
	TG=T(I)
	RG=RS  
	DO 100 J=1,20 
	 ALV1=ALV0-CPVMCL*(TG-273.15)
	 AHG=(CPD+CL*RG)*TG+ALV1*RG
	 TG=TG+(AH-AHG)/SLOPE
	 TC=TG-273.15
	 ENEW=6.112*EXP(17.67*TC/(243.5+TC))
	 RG=EPS*ENEW/(P(I)-ENEW)
  100	CONTINUE
C   
C   ***  Calculate conserved variable at top of downdraft   ***
C
	EG=RG*P(I)/(EPS+RG)
	SPD=CPD*LOG(TG)-RD*LOG(P(I)-EG)+
     1    ALV1*RG/TG
	TVD(I)=TG*(1.+RG/EPS)/(1.+RG)-T(I)*(1.+R(I)/EPS)/
     1    (1.+R(I))
	IF(P(I).LT.100.0)TVD(I)=0.0
	RGD0=RG
	TGD0=TG
C
C   ***   Find lifted condensation pressure     ***
C
	RH=R(I)/RS
	RH=MIN(RH,1.0)
	CHI=T(I)/(1669.0-122.0*RH-T(I))
	PLCL=1.0
	IF(RH.GT.0.0)THEN
	 PLCL=P(I)*(RH**CHI)
	END IF
C
C   ***  Begin updraft loop   ***
C
	SUM=0.0
	RG0=R(I)
	TG0=T(I)
	DO 200 J=I,N
C
C   ***  Calculate estimates of the rates of change of the entropies  ***
C   ***           with temperature at constant pressure               ***
C  
	 RS=EPS*ES(J)/(P(J)-ES(J))
	 ALV=ALV0-CPVMCL*(T(J)-273.15)
	 SL=(CPD+R(I)*CL+ALV*ALV*RS/(RV*T(J)*T(J)))/T(J)
	 SLP=(CPD+RS*CL+ALV*ALV*RS/(RV*T(J)*T(J)))/T(J)
C   
C   ***  Calculate lifted parcel temperature below its LCL   ***
C
	 IF(P(J).GE.PLCL)THEN
	  TLR(I,J)=T(I)*(P(J)/P(I))**(RD/CPD)
	  TLP(I,J)=TLR(I,J) 
	  LW(I,J)=0.0
	  TLVR(I,J)=TLR(I,J)*(1.+R(I)/EPS)/(1.+R(I))
	  TLVP(I,J)=TLVR(I,J)
	  TVRDIF(I,J)=TLVR(I,J)-T(J)*(1.+R(J)/EPS)/(1.+R(J))
	  TVPDIF(I,J)=TVRDIF(I,J)
	 ELSE
C
C   ***  Iteratively calculate lifted parcel temperature and mixing   ***
C   ***    ratios for both reversible and pseudo-adiabatic ascent     ***
C
	 TG=T(J)
	 RG=RS
	 DO 150 K=1,20
	  EM=RG*P(J)/(EPS+RG)
	  ALV=ALV0-CPVMCL*(TG-273.15)
	  SG=(CPD+R(I)*CL)*LOG(TG)-RD*LOG(P(J)-EM)+
     1      ALV*RG/TG
	  TG=TG+(S-SG)/SL  
	  TC=TG-273.15
	  ENEW=6.112*EXP(17.67*TC/(243.5+TC))
	  RG=EPS*ENEW/(P(J)-ENEW)           
  150	 CONTINUE
	 TLR(I,J)=TG
	 TLVR(I,J)=TG*(1.+RG/EPS)/(1.+R(I))
	 LW(I,J)=R(I)-RG
	 LW(I,J)=MAX(0.0,LW(I,J))
	 TVRDIF(I,J)=TLVR(I,J)-T(J)*(1.+R(J)/EPS)/(1.+R(J))
C
C   ***   Now do pseudo-adiabatic ascent   ***
C
	 TG=T(J)
	 RG=RS
	 DO 180 K=1,20 
	  CPW=0.0
	  IF(J.GT.1)THEN
	   CPW=SUM+CL*0.5*(RG0+RG)*(LOG(TG)-LOG(TG0))
	  END IF
	  EM=RG*P(J)/(EPS+RG)
	  ALV=ALV0-CPVMCL*(TG-273.15)
	  SPG=CPD*LOG(TG)-RD*LOG(P(J)-EM)+CPW+
     1      ALV*RG/TG
	  TG=TG+(SP-SPG)/SLP  
	  TC=TG-273.15
	  ENEW=6.112*EXP(17.67*TC/(243.5+TC))
	  RG=EPS*ENEW/(P(J)-ENEW)           
  180	 CONTINUE
	 TLP(I,J)=TG
	 TLVP(I,J)=TG*(1.+RG/EPS)/(1.+RG)
	 TVPDIF(I,J)=TLVP(I,J)-T(J)*(1.+R(J)/EPS)/(1.+R(J))
	 RG0=RG
	 TG0=TG
	 SUM=CPW
 	END IF
  200	CONTINUE
	IF(I.EQ.1)GOTO 500
C
C   ***  Begin downdraft loop   ***
C
	SUM2=0.0
	DO 300 J=I-1,1,-1
C
C   ***  Calculate estimate of the rate of change of entropy          ***
C   ***           with temperature at constant pressure               ***
C  
	 RS=EPS*ES(J)/(P(J)-ES(J))
	 ALV=ALV0-CPVMCL*(T(J)-273.15)
	 SLP=(CPD+RS*CL+ALV*ALV*RS/(RV*T(J)*T(J)))/T(J)
	 TG=T(J)
	 RG=RS
C
C   ***  Do iteration to find downdraft temperature   ***
C
	 DO 250 K=1,20
	  CPW=SUM2+CL*0.5*(RGD0+RG)*(LOG(TG)-LOG(TGD0))
	  EM=RG*P(J)/(EPS+RG)
	  ALV=ALV0-CPVMCL*(TG-273.15)
	  SPG=CPD*LOG(TG)-RD*LOG(P(J)-EM)+CPW+
     1      ALV*RG/TG
	  TG=TG+(SPD-SPG)/SLP  
	  TC=TG-273.15
	  ENEW=6.112*EXP(17.67*TC/(243.5+TC))
	  RG=EPS*ENEW/(P(J)-ENEW)           
  250	 CONTINUE
	 SUM2=CPW
	 TGD0=TG
	 RGD0=RG
	 TLP(I,J)=TG
	 TLVP(I,J)=TG*(1.+RG/EPS)/(1.+RG)
	 TVPDIF(I,J)=TLVP(I,J)-T(J)*(1.+R(J)/EPS)/(1.+R(J))
	 IF(P(I).LT.100.0)TVPDIF(I,J)=0.0
	 TVPDIF(I,J)=MIN(TVPDIF(I,J),0.0)
	 TLR(I,J)=T(J)
	 TLVR(I,J)=T(J)
	 TVRDIF(I,J)=0.0
	 LW(I,J)=0.0
  300	CONTINUE
  500   CONTINUE
C
C  ***  Begin loop to find CAPE, PA, and NA from reversible and ***
C  ***            pseudo-adiabatic ascent, and DCAPE            ***
C
	DO 800 I=1,N
	 CAPER(I)=0.0
	 CAPEP(I)=0.0
	 DCAPE(I)=0.0
	 PAP(I)=0.0
	 PAR(I)=0.0
	 NAP(I)=0.0
	 NAR(I)=0.0
C
C   ***   Find lifted condensation pressure     ***
C
	RS=EPS*ES(I)/(P(I)-ES(I))
	RH=R(I)/RS
	RH=MIN(RH,1.0)
	CHI=T(I)/(1669.0-122.0*RH-T(I))
	PLCL=1.0
	IF(RH.GT.0.0)THEN
	 PLCL=P(I)*(RH**CHI)
	END IF
C
C   ***  Find lifted condensation level and maximum level   ***
C   ***               of positive buoyancy                  ***
C
	 ICB=N
	 INBR=1
	 INBP=1
	 DO 550 J=N,I,-1
	  IF(P(J).LT.PLCL)ICB=MIN(ICB,J)
	  IF(TVRDIF(I,J).GT.0.0)INBR=MAX(INBR,J)
	  IF(TVPDIF(I,J).GT.0.0)INBP=MAX(INBP,J)
  550	 CONTINUE
	  IMAX=MAX(INBR,I)
	   DO 555 J=IMAX,N
	    TVRDIF(I,J)=0.0
  555	   CONTINUE
	  IMAX=MAX(INBP,I)
	   DO 565 J=IMAX,N
	    TVPDIF(I,J)=0.0
  565	   CONTINUE
C
C   ***  Do updraft loops        ***
C
	 IF(INBR.GT.I)THEN
	  DO 600 J=I+1,INBR
	   TVM=0.5*(TVRDIF(I,J)+TVRDIF(I,J-1))
	   PM=0.5*(P(J)+P(J-1))
	   IF(TVM.LE.0.0)THEN
	    NAR(I)=NAR(I)-RD*TVM*(P(J-1)-P(J))/PM
	   ELSE
	    PAR(I)=PAR(I)+RD*TVM*(P(J-1)-P(J))/PM
	   END IF
  600	  CONTINUE
	  CAPER(I)=PAR(I)-NAR(I)
	 END IF
	 IF(INBP.GT.I)THEN
	  DO 650 J=I+1,INBP
	   TVM=0.5*(TVPDIF(I,J)+TVPDIF(I,J-1))
	   PM=0.5*(P(J)+P(J-1))
	   IF(TVM.LE.0.0)THEN
	    NAP(I)=NAP(I)-RD*TVM*(P(J-1)-P(J))/PM
	   ELSE
	    PAP(I)=PAP(I)+RD*TVM*(P(J-1)-P(J))/PM
	   END IF
  650	  CONTINUE
	  CAPEP(I)=PAP(I)-NAP(I)
	 END IF
C
C  ***       Find DCAPE     ***
C
	 IF(I.EQ.1)GOTO 800
	 DO 700 J=I-1,1,-1
	  TVDIFM=TVPDIF(I,J+1)
	  IF(I.EQ.(J+1))TVDIFM=TVD(I)
	  TVM=0.5*(TVPDIF(I,J)+TVDIFM)
	  PM=0.5*(P(J)+P(J+1))
	  IF(TVM.LT.0.0)THEN
	   DCAPE(I)=DCAPE(I)-RD*TVM*(P(J)-P(J+1))/PM
	  END IF
  700	 CONTINUE	  	
  800	CONTINUE

	 cape = capep(1)

	return
	END
