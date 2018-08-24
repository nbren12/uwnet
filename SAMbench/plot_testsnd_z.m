clear all; define_constants

fig0 = 40;
caseid = 'testsnd';
% $$$ caseid = 'KWAJEX';
% $$$ caseid = 'DYCOMS_RF02';

plottag = sprintf('%s: 2D',caseid);
filetag = caseid;

cd OUT_STAT/
d = list_matching_files_cell(caseid,'2d','.nc');
cd ..

    for n = 1:length(d)
tag = d{n};
out(n).nc = sprintf('OUT_STAT/%s',tag);

tag_trim = tag(length(caseid)+2:end-3);
ind = find(tag_trim=='_');
tag_trim(ind) = ' ';

out(n).name = sprintf('%s',tag_trim);
disp([tag ' ' tag_trim])
names{n} = tag_trim;
    end

    frac_clb = 0.5;
    disp(['Cloud base is the lowest level at which the cloud fraction'])
    disp(['  achieves ' sprintf('%d',100*frac_clb) ' percent of its ' ...
          'maximum value']);

    for q = 1:length(out)
      wh = {'time','z','p'};
      for m = 1:length(wh)
        out(q).(wh{m}) = double(ncread(out(q).nc,wh{m}));
      end
      out(q).zkm = out(q).z/1000;

      Nt = length(out(q).time);

      whz = {'CLD', ...
               'RELH','U','V','W2','TL','QV','QCL','QCI','QPL','QPI', ...
             'QS','QSMPHY','QSSED','QSSDFL','QSADV','QSDIFF','TAUQS','QSOEFFR','QSFLXR', ...
               'QTFLUX','TLFLUX','BUOYA','TL2','PRECIP', ...
               'RADLWUP','RADLWDN','RADSWUP','RADSWDN', ...
               'RADQR','RADQRLW','RADQRSW','RADQRCLW','RADQRCSW','WOBS','RHO'};
      for m = 1:length(whz)
        tmp = double(ncread(out(q).nc,whz{m}));
        if size(tmp,1)==Nt
          out(q).(whz{m}) = mean(tmp);
        elseif size(tmp,2)==Nt
          out(q).(whz{m}) = mean(tmp,2);
        else
          error(sprintf('One of the dimensions of %s should be Nt.',whz{m}));
        end
      end
      out(q).QT = out(q).QV + out(q).QCL;

    end

    stufftoplot = whz;

    nfig = 1;
    for kk = 1:length(stufftoplot)
      if mod(kk,3) == 1
        figure(fig0+nfig); clf
        nfig = nfig + 1;
        nsub = 1;
      else
        nsub = nsub + 1;
      end
      hLa = subplot(2,2,nsub);
      hL1 = plot_pldataz(stufftoplot{kk},'z, km',plottag,-1,out,'zkm', ...
                        stufftoplot{kk});
      if kk==length(stufftoplot) | mod(kk,3)==0
        nsub = nsub + 1;
        hLb = subplot(2,2,nsub);
        hL2 = plot_pldataz(stufftoplot{kk},'z, km','',-1,out,'zkm', ...
                          stufftoplot{kk});
        hLL = legend(hL2,names);
        set(hLL,'FontSize',8,'Location','NorthWest');
        set(hLb,'Visible','off');
        set(hL2,'Visible','off');
      end
      eval(sprintf('print -dpng -r200 %s_diagnostic_plot%.2d.png',filetag,nfig))
    end

