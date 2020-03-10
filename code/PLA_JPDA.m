% PLA-JPDA for multi-person tracking
%% PL-based multipath suppression
load('../result/crslt1.mat');
crslt = crslt(46:245,:);  % obtain the valid signal
P = crslt.^2;
[r,c] = size(P);
ht = P;
cm = zeros(size(P));
gamma = 2.8;  % a exponent related to the environment
distWhole = (1:c);
dr = 1/156; % distance resolution
for i = 1:r
    [pk,loc] = max(P(i,:));
    thrs = 0;

    while pk>0
        cm(i,loc) = 1;
        ht(i,loc) = 0;
        indx = find(ht(i,:)>0);
        if isempty(indx)
            break;
        end
        thrs = pk.*((distWhole*dr+0.5)./(loc*dr+0.5)).^(-gamma.*sign(distWhole-loc));

        for j = 1:length(indx)
            if ht(i,indx(j)) < thrs(indx(j))
                ht(i,indx(j)) = 0;
            end
        end
        [pk, loc] = max(ht(i,:));
    end
end

for i = 1:r
    dist1 = [];
    for j = 1:length(cm(i,:))
        if cm(i,j) > 0
            dist1=[dist1,(j * dr + 0.5)];
        end
    end
    dist{i}=dist1;
end


%% Adaptive JPDA
len = length(dist);
fps =20;
n=len;
c=2;T=1;
target_position=[1.3 0; 3.8 0];  % initial point
Pd=0.9;  % probability of detection
g_sigma=0.25;  % threshold of validation gate
target_delta=[1 1; 1 1];
P=zeros(2,2,c);  % covariance matrix
P1=[target_delta(1,1)^2 0; 0 target_delta(1,2)^2 ];
P2=[target_delta(2,1)^2 0; 0 target_delta(2,2)^2 ];
P(:,:,1)=P1;
P(:,:,2)=P2;
F = [1 T ;0 1];  % state transition matrix
H = [1 0];  % observation matrix
R=[1 1]; % measurement noise
Q=0.01;  % process noise
G=[T^2/2;T];
x_filter=zeros(2,c,n);  % store the filtered data

S=zeros(1,1,c);
Z_predic=zeros(1,2);  % observation
x_predic=zeros(2,1);  % state
gate_volume=zeros(1,2);
y0=[];
Cflag = zeros(1,n);  % flag indicating whether the persons are close to each other
for t=1:n
    y=[];
    y1=[];
    b=zeros(1,2);
    for count=1:length(dist{t})
        b(1)=t;
        b(2)=dist{t}(count);
        y1=[y1 b'];
    end
    y0=[y0,y1];  % all the measurements
    
    % adaptive threshold
    if count>1
        maxdiff = max(y1(2,:))-min(y1(2,:));
        if  maxdiff < 0.5
            Cflag(1,t) = 1;
            g_sigma = max([(maxdiff).^2, 0.09]);
        else
            g_sigma = 0.25;
        end
    end
    
    for i=1:c
        if t~=1
            x_predic(:,i) = F*x_filter(:,i,t-1);
        else
            x_predic(:,i)=target_position(i,:)';
        end
        P_predic=F*P(:,:,i)*F'+G*Q*G';  % update the covariance matrix of x_predic
        Z_predic(:,i)=H*x_predic(:,i);
        S(:,:,i)=H*P_predic*H'+R(i);
        gate_volume(i)=2*sqrt(g_sigma); % area of the validation gate
    end
    
    
    % Generate observation confirmation matrix, Q2
    m1=0;
    [n1,n2]=size(y1);
    Q1=zeros(200,3);
    for j=1:n2
        flag=0;
        for i=1:c
            d=y1(2,j)-Z_predic(:,i);
            D = d'*d;
            if D<=g_sigma
                flag=1;
                Q1(m1+1,1)=1;
                Q1(m1+1,i+1)=1;
            end
        end
        if flag==1
            y=[y y1(2,j)];
            m1=m1+1;  % number of valid observations
        end
    end
    Q2=Q1(1:m1,1:3);
    

    % Generate interconnect matrix, A_matrix
    A_matrix=zeros(m1,3,1000);
    A_matrix(:,1,1:1000)=1;
    if m1~=0 
        num=1;
        for i=1:m1
            if Q2(i,2)==1
                A_matrix(i,2,num)=1;A_matrix(i,1,num)=0;    % assign i to track1
                num=num+1;
                for j=1:m1
                    if (i~=j)&&(Q2(j,3)==1)    % assign i to track1, j to track2
                        A_matrix(i,2,num)=1;A_matrix(i,1,num)=0;
                        A_matrix(j,3,num)=1;A_matrix(j,1,num)=0;
                        num=num+1;
                    end
                end
            end
        end
        
        for i=1:m1
            if Q2(i,3)==1
                A_matrix(i,3,num)=1;A_matrix(i,1,num)=0;    % assign i to track2
                num=num+1;
            end
        end
    end
    if m1==0
        if t~=1
            x_filter(:,:,t)=F*x_filter(:,:,t-1);
        else
            x_filter(:,:,t)=target_position';
        end
        continue;
    end
    A_matrix=A_matrix(:,:,1:num);
    
    
    % Calculate the assosiation probability
    Pr=zeros(1,num);
    for i=1:num
        False_num=m1;
        N=1;
        for j=1:m1
            mea_indicator=sum(A_matrix(j,2:3,i));
            if mea_indicator==1
                False_num=False_num-1;
                if A_matrix(j,2,i)==1  %Observation associated with target 1
                    b=(y(:,j)-Z_predic(:,1))'*inv(S(:,:,1))*(y(:,j)-Z_predic(:,1));
                    N=N/sqrt(det(2*pi*S(:,:,1)))*exp(-1/2*b);
                else  %Observation associated with target 2
                    b=(y(:,j)-Z_predic(:,2))'*inv(S(:,:,2))*(y(:,j)-Z_predic(:,2));
                    N=N/sqrt(det(2*pi*S(:,:,2)))*exp(-1/2*b);
                end
            end
        end
        if Pd==1
            a=1;
        else
            a=1;
            for j=1:c
                target_indicator=sum(A_matrix(:,j+1,i));
                a=a*Pd^target_indicator*(1-Pd)^(1-target_indicator);
            end
        end
        V=gate_volume(1)+gate_volume(2);
        
        a1=1;
        for j=1:False_num
            a1=a1*j;
        end
        Pr(i)=N*a*a1/(V^False_num);
    end
    Pr=Pr/sum(Pr);
    
    
    % Calculate mixture weights
    U=zeros(m1+1,c);
    for i=1:c
        for j=1:m1
            for k=1:num
                U(j,i)=U(j,i)+Pr(k)*A_matrix(j,i+1,k);
            end
        end
    end
    U(m1+1,:)=1-sum(U(1:m1,1:c),1);
    
    Aflag = ones(1,c);
    if m1==1
        for i = 1:c
            if U(1,i)<0.5
                Aflag(1,i) = 0;
            end
        end
    else  % modify mixture weight
        for i=1:c
            for j=1:m1
                if U(j,i) < 1/m1
                    U(j,i)=0;
                end
            end
        end
        for i=1:c
            U(:,i) = U(:,i)./sum(U(:,i));
        end
    end
    
    
    % Update predictions and estimates
    for i=1:c  
        if Aflag(1,i)
            P_predic = F*P(:,:,i)*F'+G*Q*G';
            K (:,:,i)= P_predic*H'/(S(:,:,i));
            P(:,:,i)= P_predic-(1-U(m1+1,i))*K(:,:,i)*S(:,:,i)*K(:,:,i)';
        end
    end
    for i=1:c
        if Aflag(1,i)
            a=0;
            b=0;
            x_filter2=0;
            for j=1:m1
                x_filter2=x_filter2+U(j,i)*(x_predic(:,i)+ K(:,:,i)*(y(:,j)- Z_predic(:,i)));
            end
            x_filter2=U(m1+1,i)*x_predic(:,i)+x_filter2;
            x_filter(:,i,t)=x_filter2;
            for j=1:m1+1
                if j==m1+1
                    a=x_predic(:,i);
                else
                    a=x_predic(:,i)+ K (:,:,i)*(y(:,j)- Z_predic(:,i));
                end
                b=b+U(j,i)*(a*a'-x_filter2*x_filter2');
            end
            P(:,:,i)=P(:,:,i)+b;
            if P(1,1,i)>10   % prevent P from infinitely increasing when there are no associated points
                P(:,:,i) = P1;
            end
        else
            if t~=1
                if Cflag(1,t)
                    x_filter(:,i,t)=x_filter(:,i,t-1);
                else
                    x_filter(:,i,t)=F*x_filter(:,i,t-1);
                end
            else
                x_filter(:,i,t)=target_position(i,:)';
            end
        end
    end
end

%% Output
for j=1:n
pa(j)=x_filter(1,1,j);
pb(j)=x_filter(1,2,j);
end
r1 = [pa',pb'];
% Store r1, r2 and r3 in ../result/3GroupDist.mat