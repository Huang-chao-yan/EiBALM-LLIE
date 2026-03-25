function f = FFTsolution(dx, dy, v, lambda, mu)
% the solution of min{1/2*int(gradient u-d)^2+lambda/2*int(u-v)^2+mu/2*int u^2}
% by chaoyan
sizev = size(v);
otfDx = psf2otf([1, -1], sizev);
otfDy = psf2otf([1; -1], sizev);
Nomin = lambda*fft2(v)+dx;
Denom = (abs(otfDx).^2 + abs(otfDy ).^2)+lambda+mu;
s = Nomin./Denom;
f = real(ifft2(s));
end