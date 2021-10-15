a = [1 2 1 2 1]'; 
b = [2 1 2 1 1]';
c = [2 1 2 2 1]';

ab=mi(a,b);

ha=h(a);
hab=MIToolboxMex(6,a,b);
MI_1=ha-hab;
mi_1=ha-ch(a,b);

hb=h(b);
hba=ch(b,a);
mi_2=hb-hba;


%H(X,)
h_ab=h([a,b]);
mi_3=ha+hb-h_ab;

% disp([ab,MI_1,mi_1,mi_2,mi_3]);

% disp([cmi(a,b,c),cmi(b,a,c)]);

% disp([mi3(a,b,c),mi3(b,a,c),mi3(c,a,b)]);

S=[1,0,1,1;
  0,1,1,1 ;
  1,1,0,1	;
  0,0,0,1 ];
mi(S(:,1),S(:,3))
mi(S(:,2),S(:,3))
mi(S(:,1),S(:,2))
cmi(S(:,1),S(:,2),S(:,3))
cmi(S(:,1),S(:,2),S(:,4))
