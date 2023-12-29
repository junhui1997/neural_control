iiwa = importrobot('iiwa14.urdf');
iiwa.DataFormat = 'column';
startConfig = [0 -pi/4 pi 3*pi/2 0 -pi/2 pi/8]';
goalConfig = [0 -pi/4 pi 3*pi/4 0 -pi/2 pi/8]';
q = trapveltraj([startConfig goalConfig],100,'EndTime',3);
isConfigInCollision = false(100,1);
configCollisionPairs = cell(100,1);
sepDistForConfig = zeros(iiwa.NumBodies+1,iiwa.NumBodies+1,100);
for i = 1:length(q)
    [isColliding, sepDist] = checkCollision(iiwa,q(:,i),'Exhaustive','on','SkippedSelfCollisions','parent');
    isConfigInCollision(i) = isColliding;
    sepDistForConfig(:,:,i) = sepDist;
end
for i = 1:length(q)
    sepDist = sepDistForConfig(:,:,i);
    [body1Idx,body2Idx] = find(isnan(sepDist));

    collidingPairs = unique(sort([body1Idx,body2Idx],2));
    configCollisionPairs{i} = collidingPairs;
end
any(isConfigInCollision)
firstCollisionIdx = find(isConfigInCollision,1);

% Visualize the first configuration that is in collision.
figure;
show(iiwa,q(:,firstCollisionIdx));
exampleHelperHighlightCollisionBodies(iiwa,configCollisionPairs{firstCollisionIdx}+1,gca);