function [out] = MakeRowVector(in)

if length(find(size(in)>1))>1
  error('input must only have a single dimension with length > 1')
end

out = reshape(in,[1 length(in)]);
