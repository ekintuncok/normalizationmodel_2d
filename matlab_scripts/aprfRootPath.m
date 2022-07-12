function rootPath = aprfRootPath()
% Return the path to the project attention PRF
%
% This function must reside in the directory at the base of the
% forward modeling code directory structure.  It is used to determine the location of various
% sub-directories.
% 
% Example:
%   cd(aprfRootPath)

rootPath=which('aprfRootPath');

rootPath=fileparts(rootPath);

return
