% function f = FFTsolution(dx, dy, v, lambda, mu)
% % the solution of min{1/2*int(gradient u-d)^2+lambda/2*int(u-v)^2+mu/2*int u^2}
% % by chaoyan
% sizev = size(v);
% otfDx = psf2otf([1, -1], sizev);
% otfDy = psf2otf([1; -1], sizev);
% Nomin = lambda*fft2(v)+dx;
% Denom = (abs(otfDx).^2 + abs(otfDy ).^2)+lambda+mu;
% s = Nomin./Denom;
% f = real(ifft2(s));
% end

function f = FFTsolution(dx, dy, v, lambda, mu)
% Solve, with periodic boundary conditions:
%
% min_u  0.5*||Dx(u)-dx||_F^2 ...
%      + 0.5*||Dy(u)-dy||_F^2 ...
%      + lambda/2*||u-v||_F^2 ...
%      + mu/2*||u||_F^2
%
% dx, dy, v are spatial-domain, real-valued 2-D arrays.
% Assumes lambda >= 0, mu >= 0, and lambda + mu > 0.

    sizev = size(v);

    otfDx = psf2otf([1, -1], sizev);
    otfDy = psf2otf([1; -1], sizev);

    Nomin = conj(otfDx).*fft2(dx) ...
          + conj(otfDy).*fft2(dy) ...
          + lambda*fft2(v);

    Denom = abs(otfDx).^2 ...
          + abs(otfDy).^2 ...
          + lambda + mu;

    f = real(ifft2(Nomin./Denom));
end