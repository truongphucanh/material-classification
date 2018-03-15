fid = fopen('FMD_original.txt');
while ~feof(fid)
    from = fgetl(fid);
    to = strrep(from,'original','texture');
    disp(to);
    original = imread(from);
    texture = rangefilt(original);
    imwrite(texture, to);
end