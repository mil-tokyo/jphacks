angular.module('Example', ['LocationBar'])
	.run(['$rootScope', function (scope) {
				scope.example = 'Parent';

				scope.example2 = 'Parent Only';

			}])
	.controller('MainCtrl', ['$scope', function (scope) {
				scope.example = 'Child';
				scope.aside = {
					"title": "Title",
					"content": "Hello Aside<br />This is a multiline message!"
				};

				scope.square = function (n) {
					return n * n;
				};
			}]);
