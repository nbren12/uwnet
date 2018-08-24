begin

f=addfile("../../OUT_STAT/SOLAR_32x28sp_4km_20s_M5RRTM_C100_280.nc","r")

nav=3

print(f->time+"   "+f->PW+"    "+f->PREC+"   "+(f->SST+300.))
nt=dimsizes(f->time)-1
nz=dimsizes(f->z)
LHF = f->LHF(nt-nav:nt)
SHF = f->SHF(nt-nav:nt)
swnt = f->SWNS(nt-nav:nt)
lwnt = -f->LWNS(nt-nav:nt)
swntt = f->SWNT(nt-nav:nt)
lwntt = -f->LWNT(nt-nav:nt)
print("swns="+avg(swnt)+" lwns= "+"  "+avg(lwnt)+" lhf= "+avg(LHF)+" shf= "+avg(SHF))
print("swnt="+avg(swntt)+" lwnt= "+"  "+avg(lwntt))
print("sw absorbed:"+avg(swntt-swnt)+" lw absorbed: "+"  "+avg(lwntt-lwnt))
print("balance at sfc:"+avg(swnt+lwnt-LHF-SHF))
print("balance at top:"+avg(swntt+lwntt))
print("incoming solar:"+avg(f->SOLIN(nt-nav:nt)))
print("SST:"+avg(f->SST(nt-nav:nt)))

;print(f->z+"      "+f->p+"   "+f->TABS(nt-1,:)*(1000./f->p)^(287./1004.)+"    "+f->QT(nt,:)+"     "+0.+"      "+0.)
end
